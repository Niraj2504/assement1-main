import io
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
import csv

load_dotenv()

# Initialize the model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Create LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
LLMapp = workflow.compile(checkpointer=memory)

app = FastAPI() # Its used to create a server which serves Http requests

# Mount static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SHARED_THREAD_ID = "shared-context"
ADMIN_MESSAGES = []  # holds latest admin context messages

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_user(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/query")
async def admin_query_model(request: Request):
    global ADMIN_MESSAGES
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            raise ValueError("Missing query")

        config = {"configurable": {"thread_id": SHARED_THREAD_ID}}
        input_messages = [HumanMessage(query)]
        output = LLMapp.invoke({"messages": input_messages}, config)
        ADMIN_MESSAGES = output["messages"]  # cache for users
        response_message = output["messages"][-1].content
        return {"response": response_message}

    except Exception as e:
        print(f"[ADMIN ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}

@app.post("/query")
async def query(
    request: Request,
    query: str = Form(...),
    session_id: str = Form(None),
    file: UploadFile = File(None),
):
    """Accepts a CSV / text / PDF / image file plus a user query, merges with admin context."""
    try:
        thread_id = f"user-{session_id or str(uuid.uuid4())}"

        # -------- Parse uploaded file --------
        file_text = ""
        if file is not None:
            raw = await file.read()
            # Handle CSV specifically for better formatting
            is_csv = (
                file.filename and file.filename.lower().endswith(".csv")
            ) or file.content_type in {"text/csv", "application/vnd.ms-excel"}

            if is_csv:
                try:
                    decoded = raw.decode("utf-8", errors="ignore")
                    reader = csv.reader(io.StringIO(decoded))
                    # Limit rows & columns to avoid huge prompts
                    rows = [", ".join(row[:10]) for _, row in zip(range(50), reader)]
                    file_text = "\n".join(rows)
                except Exception as csv_err:
                    print(f"[CSV PARSE ERROR] {csv_err}")
                    file_text = "<failed to parse CSV>"
            else:
                # naive text decode; for PDF/image integrate textract, pytesseract, etc.
                try:
                    file_text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    file_text = f"<binary {file.filename} : {len(raw)} bytes>"

        # -------- Build combined prompt --------
        prompt_parts = []
        if file_text:
            prompt_parts.append(f"[File: {file.filename}]\n{file_text}\n")
        prompt_parts.append(query)
        combined_query = "\n".join(prompt_parts)

        print(f"[QUERY] {combined_query}")

        messages = ADMIN_MESSAGES + [
            HumanMessage(combined_query), 
            SystemMessage('I have a div tag with chat-box id where I am going to show the output. In response you have to Generate a HTML/CSS/JS Note: Please give me div tag code only no html and body tag needed')
        ] if ADMIN_MESSAGES else [
            HumanMessage(combined_query), 
            SystemMessage('I have a div tag with chat-box id where I am going to show the output. In response you have to Generate a HTML/CSS/JS Note: Please give me div tag code only no html and body tag needed')
        ]
        cfg = {"configurable": {"thread_id": thread_id}}
        output = LLMapp.invoke({"messages": messages}, cfg)
        return {"response": output["messages"][-1].content}

    except Exception as e:
        print(f"[FILE QUERY ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}


@app.post("/admin/reset")
def reset_admin_context():
    global ADMIN_MESSAGES
    ADMIN_MESSAGES = []
    return {"message": "Admin context cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
