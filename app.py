import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import requests

load_dotenv()

finance_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Finance Tool',
    role="Get financial data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, stock_fundamentals=True)],
    markdown=True,
    instructions=[
        'Use tables to display data.',
        'Provide detailed financial analysis.',
        'Ensure accuracy in stock market insights.',
        'Include relevant stock trends and historical data when necessary.',
        'Explain complex financial terms in simple language.',
    ]
)

web_search_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Web Search Tool',
    role="Search Web",
    tools=[DuckDuckGo()],
    markdown=True,
    instructions=[
        'Always include sources.',
        'Provide the most relevant and recent financial news.',
        'Avoid speculation and prioritize factual information.',
    ]
)

team_leader_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Leader Tool',
    team=[web_search_agent, finance_agent],
    markdown=True,
    instructions=[
        'Always include sources.',
        'Use tables to display data.',
        'You are a financial assistant.',
        'Only respond to finance-related questions.',
        'If you don\'t know the name of a company, search for it first.',
        "If a query is not related to finance, reply: 'I specialize in financial topics. Please ask me about stock markets, investments, or financial data.'",
        'Summarize insights concisely but provide sufficient details for understanding.',
        'Use markdown formatting to make information clear and visually appealing.',
        'Always end responses with: "Do you want any more help? 😊"',
    ]
)

st.set_page_config(page_title="Chat Bot", initial_sidebar_state='expanded', page_icon='🤖')

st.title("📊 AI Financial & Web Search Assistant")

# st.sidebar.title("⚙️ Options")
# if st.sidebar.button("🗑️ Clear Chat History"):
#     st.session_state.history = []

# if "history" not in st.session_state:
#     st.session_state.history = []

st.markdown("""
### 🔍 How It Works
1. Enter your query in the chat box below.
2. Press Enter to receive financial insights or web search results.
3. The AI will format responses in markdown with tables for easy readability.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask your financial question:")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = team_leader_agent.run(prompt).content 
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

css_code = """
body {
  background-color: #f0f0f0; /* Light gray background */
}
"""

st.markdown(f"""<style>{css_code}</style>""", unsafe_allow_html=True)
