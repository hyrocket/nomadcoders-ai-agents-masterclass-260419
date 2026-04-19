import dotenv   # dotenv는 환경 변수를 로드하는 라이브러리

dotenv.load_dotenv()    # .env 파일에 저장된 환경 변수를 로드
from openai import OpenAI
import asyncio  # asyncio는 비동기 프로그래밍을 위한 라이브러리
import base64   # base64는 이미지 데이터를 인코딩/디코딩하는 라이브러리
import os
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
)

client = OpenAI()

VECTOR_STORE_ID = os.environ.get("VECTOR_STORE_ID")

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "life-coach-memory.db",
    )
session = st.session_state["session"]


async def paint_history():  # 이전 대화 기록을 화면에 표시하는 함수
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\\$"))
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 웹 검색중...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ 목표 문서 검색중...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)


asyncio.run(paint_history())


def update_status(status_container, event):   # 각 이벤트 타입에 따라 status 박스의 메시지와 상태를 업데이트
    status_messages = {
        "response.web_search_call.completed": ("✅ 웹 검색완료!", "complete"),
        "response.web_search_call.in_progress": ("🔍 웹검색을 시작합니다...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색중...", "running"),
        "response.file_search_call.completed": ("✅ 문서 검색완료!", "complete"),
        "response.file_search_call.in_progress": ("🗂️ 목표 문서 검색 시작...", "running"),
        "response.file_search_call.searching": ("🗂️ 목표 문서 검색중...", "running"),
        "response.image_generation_call.generating": ("🎨 이미지 생성중...", "running"),
        "response.image_generation_call.in_progress": ("🎨 이미지 생성중...", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):
    agent = Agent(
        name="LIFE COACH",
        model="gpt-4o-mini",
        instructions="""
        당신은 한국어로 소통하는 열정적이고 지지적인 라이프 코치입니다.
        당신의 역할은 다음과 같습니다:
        사용자가 목표를 달성할 수 있도록 동기를 부여하고 격려합니다
        실용적인 자기계발 팁과 조언을 제공합니다

        You MUST use the Web Search Tool before giving any advice:
            - 동기부여 컨텐츠, 자기개발팁, 습관형성 조언을 검색하고 이를 활용하여 조언하세요
            - Always search BEFORE answering. Never answer from memory alone.

        You MUST use the File Search Tool when:
            - 사용자가 목표나 진행 상황에 대해 물어볼 때
            - 사용자의 개인 목표 문서를 참조해야 할 때
            - 과거 기록을 바탕으로 조언할 때

        You MUST use the Image Generation Tool when:
            - 사용자가 비전 보드를 요청할 때
            - 사용자가 목표 달성을 축하할 때
            - 동기부여 포스터나 이미지가 필요할 때
            - Always generate images with English prompts for better results.

        검색 후 따뜻한 말로 답변하고, 항상 응원 메시지로 마무리합니다.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,
                }
            ),
        ],
    )

    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""

        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\\$"))

                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)


st.title("🔥Life Coach Agent")
st.caption("당신의 성장을 응원하는 AI 라이프 코치입니다!")

prompt = st.chat_input(
    "라이프코치에게 고민을 털어놓거나 말을 해보세요",
    accept_file=True,
    file_type=["txt"],
)

if prompt:
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ 파일 업로드 중...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ 파일 연결 중...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ 파일 업로드 완료!", state="complete")

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))

with st.sidebar:
    reset = st.button("대화 초기화")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))