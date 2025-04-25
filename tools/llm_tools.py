# tools/llm_tools.py
import logging
from crewai.tools import BaseTool
# REMOVED: from tools.llm_service import llm_service (No longer needed)
# ADDED: Import the LLM factory and LangChain message types
from utils.llm_factory import get_llm_instance
from langchain_core.messages import HumanMessage
# Keep error handling decorators
from utils.error_handler import handle_api_error # Assuming retry isn't needed here, but handle_api_error is good

# Configure logger (using basicConfig for simplicity in tool file, adjust if needed)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use module logger

class PainPointAnalyzerTool(BaseTool):
    name: str = "Company Pain Point Analyzer"
    description: str = (
        "Analyzes a company by name to infer potential general business pain points using the configured LLM. "
        "Input must be the company name string."
    )

    # Removed @retry decorator as LLM calls via LangChain might have internal retries,
    # but kept @handle_api_error for graceful failure. Add retry back if needed.
    @handle_api_error
    def _run(self, company_name: str) -> str:
        """Uses the configured LLM via LangChain interface to identify potential pain points."""
        tool_name = self.name # Use self.name for consistency
        logger.info(f"[Tool: {tool_name}] Executing for Company: '{company_name}'")
        if not isinstance(company_name, str) or not company_name:
            logger.error(f"[Tool: {tool_name}] Invalid input provided: {company_name}")
            return "Error: Invalid company name provided."

        try:
            # --- MODIFIED: Get LLM instance from factory ---
            logger.debug(f"[Tool: {tool_name}] Getting LLM instance from factory...")
            llm = get_llm_instance()
            if llm is None:
                logger.error(f"[Tool: {tool_name}] Failed to get LLM instance from factory.")
                # handle_api_error might catch this, but explicit check is good
                return "Error: LLM instance could not be initialized for analysis."
            logger.debug(f"[Tool: {tool_name}] Using LLM instance type: {type(llm).__name__}")
            # --- END MODIFICATION ---

            # Define the prompt (remains the same)
            prompt = (
                f"Identify 3-5 potential general business pain points for a company named '{company_name}'. "
                f"Consider areas like market competition, operational efficiency, scaling challenges, "
                f"technological adoption, financial pressures, customer retention, or talent acquisition/management. "
                f"Provide the answer as a numbered or bulleted list."
            )
            logger.debug(f"[Tool: {tool_name}] Prompt for '{company_name}': {prompt[:100]}...")

            # --- MODIFIED: Use LangChain invoke method ---
            logger.debug(f"[Tool: {tool_name}] Making LLM call via LangChain invoke for '{company_name}'...")
            # Use HumanMessage for standard chat model interaction
            response = llm.invoke([HumanMessage(content=prompt)])

            # Extract content from the response object (usually response.content for AIMessage)
            pain_points = response.content if hasattr(response, 'content') else str(response)
            # --- END MODIFICATION ---

            logger.info(f"[Tool: {tool_name}] LLM call successful for '{company_name}'.")
            logger.debug(f"[Tool: {tool_name}] LLM Response for '{company_name}':\n{pain_points}")
            return pain_points

        except Exception as e:
            # handle_api_error decorator will catch this, but specific logging here is useful
            logger.error(f"[Tool: {tool_name}] LLM call failed during LangChain invoke for '{company_name}': {e}", exc_info=True)
            # The decorator will likely return the error string, but we can return here too if needed
            return f"Error: LLM query failed during analysis via LangChain."

# Instantiate the tool (remains the same)
analyze_pain_points_tool = PainPointAnalyzerTool()
