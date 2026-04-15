import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai.tasks.conditional_task import ConditionalTask
from crewai_tools import FileReadTool
from crewai.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyDWy0VO0ICZlJUWrC4NTbkc-pimxl5EDXk"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

class ClauseValidatorTool(BaseTool):
    name: str = "Clause Validator"
    description: str = "Проверяет наличие обязательных юридических терминов в тексте."

    def _run(self, text: str) -> str:
        keywords = ["права", "обязанности", "форс-мажор", "конфиденциальность"]
        found = [word for word in keywords if word.lower() in text.lower()]
        return f"Найдено обязательных разделов: {len(found)} из {len(keywords)}."

st.set_page_config(page_title="MAS Contract Analyzer", layout="wide")

with st.sidebar:
    st.header("Конфигурация системы")
    role_auditor = st.text_input("Role", "Compliance Officer")
    goal_auditor = st.text_input("Goal", "Сверка договора с регламентом")
    backstory_auditor = st.text_area("Backstory", "Вы эксперт по внутренним правилам университета.")

st.title("Прикладная МАС: Проверка договоров на практику")

col1, col2 = st.columns(2)
with col1:
    contract_file = st.file_uploader("Загрузить договор", type=["pdf", "txt"])
    knowledge_input = st.text_area("База знаний", "Регламент: Срок практики от 2 до 4 недель...")

if st.button("Запустить CrewAI") and contract_file:
    with open("contract.pdf", "wb") as f:
        f.write(contract_file.getbuffer())

    reader_tool = FileReadTool(file_path="contract.pdf")
    validator_tool = ClauseValidatorTool()

    extractor = Agent(
        role="Data Extractor",
        goal="Извлечь основные условия договора",
        backstory="Вы специалист по анализу юридической документации.",
        llm=llm,
        tools=[reader_tool],
        allow_delegation=False
    )

    auditor = Agent(
        role=role_auditor,
        goal=goal_auditor,
        backstory=backstory_auditor,
        llm=llm,
        tools=[validator_tool],
        allow_delegation=False
    )

    risk_manager = Agent(
        role="Risk Manager",
        goal="Оценить критические риски",
        backstory="Вы ведущий юрист университета.",
        llm=llm,
        allow_delegation=False
    )

    task1 = Task(
        description="Прочитай договор и выдели сроки и стороны.",
        expected_output="Список ключевых параметров договора.",
        agent=extractor
    )

    task2 = Task(
        description=f"Проверь данные, используя базу знаний: {knowledge_input}",
        expected_output="Отчет. Если есть нарушения, начни текст со слова CRITICAL.",
        agent=auditor,
        context=[task1]
    )

    task3 = ConditionalTask(
        description="Сформулируй правки для устранения рисков.",
        expected_output="Список юридических правок.",
        condition=lambda output: "CRITICAL" in output.raw,
        agent=risk_manager
    )

    task4 = Task(
        description="Сформировать финальный вердикт для подписи.",
        expected_output="Полное экспертное заключение.",
        agent=auditor,
        human_input=True
    )

    crew = Crew(
        agents=[extractor, auditor, risk_manager],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        memory=False,
        verbose=True,
        max_rpm=1
    )

    with st.spinner("Система работает. Проверьте терминал для подтверждения."):
        final_result = crew.kickoff()

    st.success("Выполнено")
    st.subheader("Результат:")
    st.write(final_result.raw)