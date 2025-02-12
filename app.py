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
    ]
)

st.set_page_config(page_title="Chat Bot", initial_sidebar_state='expanded', page_icon='🤖')

st.title("📊 AI Financial & Web Search Assistant")

st.sidebar.title("⚙️ Options")
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
### 🔍 How It Works
1. Enter your query in the text box below.
2. Click **Send** to receive financial insights or web search results.
3. The AI will format responses in markdown with tables for easy readability.
""")

query = st.text_input("💬 Enter your query:")

if st.button("🚀 Send"):
    response = team_leader_agent.run(query).content
    if response:
        st.session_state.history.append(("You", query))
        st.session_state.history.append(("🤖 AI", response))

st.markdown("""### 💬 Chat History:""")
for sender, msg in st.session_state.history:
    st.markdown(f"**{sender}:** {msg}")

css_code = """
body {
  background-color: #f0f0f0; /* Light gray background */
}
"""

st.markdown(f"""<style>{css_code}</style>""", unsafe_allow_html=True)
