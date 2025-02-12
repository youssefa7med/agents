import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create agents
finance_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Finance Tool',
    role="Get financial data",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, stock_fundamentals=True)],
    markdown=True,
    instructions=['Use tables to display data']
)

web_search_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Web Search Tool',
    role="Search Web",
    tools=[DuckDuckGo()],
    markdown=True,
    instructions=['Always include sources']
)

team_leader_agent = Agent(
    model=Groq(id='deepseek-r1-distill-llama-70b'),
    name='Leader Tool',
    team=[web_search_agent, finance_agent],
    markdown=True,
    instructions=['Always include sources', 'Use tables to display data', 'You are a helpful assistant']
)

# Streamlit UI
st.title("📊 AI Financial & Web Search Assistant")

st.markdown("""
### 🔍 How It Works
1. Enter your query in the text box below.
2. Click **Get Response** to receive financial insights or web search results.
3. The AI will format responses in markdown with tables for easy readability.
""")

query = st.text_input("💬 Enter your query:")

if st.button("🚀 Get Response"):
    response = team_leader_agent.run(query)
    if response:
        st.markdown("""### 🤖 Response:""")
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.warning("No response received. Please check your query or try again.")
