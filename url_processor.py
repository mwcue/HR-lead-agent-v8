# url_processor.py
import logging
import ast
import re
# --- CHANGE START ---
# Import CrewOutput to check its type
from crewai import Crew, Process, CrewOutput
# --- CHANGE END ---
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

def parse_url_list(agent_output: str) -> list:
    """
    Extracts URLs from agent output string, assuming Python list format or simple list.

    Args:
        agent_output: String output from agent containing URLs

    Returns:
        List of extracted URLs
    """
    # Keep the logger name consistent if desired, or use the module logger
    # logger = logging.getLogger(__name__) # Redundant if module logger is used
    logger.debug(f"Parsing URL list from raw agent output string.") # Adjusted log message
    urls = []

    # --- CHANGE START: Check input type FIRST ---
    if not isinstance(agent_output, str):
        logger.warning(f"URL Parser received non-string input: {type(agent_output)}. Cannot parse.")
        return urls
    # --- CHANGE END ---

    try: # Try ast.literal_eval
        # Clean potential "Final Answer:" prefix if present
        if agent_output.strip().upper().startswith("FINAL ANSWER:"):
            agent_output = agent_output.split(":", 1)[1].strip()
        # Attempt parsing
        potential_list = ast.literal_eval(agent_output)
        if isinstance(potential_list, list):
            urls = [str(item).strip() for item in potential_list if isinstance(item, str) and item.strip().startswith('http')]
            logger.info(f"Parsed URL list via ast: {len(urls)} URLs.")
            return urls
    except Exception as e:
        logger.warning(f"Could not parse URL list via ast: {e}")

    try: # Fallback Regex
        urls = re.findall(r'https?://[^\s\'\"\]\[\<\>]+', agent_output)
        urls = [url.strip('.,)("') for url in urls]
        # Keep the filter for common non-page extensions
        urls = [url for url in urls if not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.css', '.js', '.svg', '.webp', '.pdf'))]
        # Ensure uniqueness while preserving order
        urls = list(dict.fromkeys(urls))
        logger.info(f"Parsed URL list via regex: {len(urls)} URLs.")
        return urls
    except Exception as e:
        logger.error(f"Error during regex URL parsing: {e}")
        return []

def perform_search(agents, tasks):
    """
    Execute search tasks to find relevant HR tech URLs.

    Args:
        agents: Dictionary of initialized agents
        tasks: List of search-related tasks

    Returns:
        List of URLs found during search
    """
    url_list = []
    try:
        logger.info("--- Kicking off Search Tasks ---")

        # Set up search crew
        search_crew = Crew(
            agents=[agents['research']],
            tasks=tasks,
            process=Process.sequential,
            verbose=False # Set verbose=False for cleaner production logs unless debugging Crew steps
        )

        # Execute search
        search_results_object = search_crew.kickoff() # Rename variable
        logger.info("--- Search Tasks Finished ---")

 # --- ADDED: Log raw output from the planning task if possible ---
        # Note: This relies on accessing task outputs which can sometimes be complex in CrewAI.
        # We log the FINAL raw output, which might contain intermediate steps including the queries.
        # A more robust way might involve a custom callback handler, but let's try this first.
        final_raw_output_for_logging = "Could not extract raw output for query logging."
        if search_results_object:
             if isinstance(search_results_object, CrewOutput):
                 final_raw_output_for_logging = search_results_object.raw
             elif isinstance(search_results_object, str):
                 final_raw_output_for_logging = search_results_object

        # Log the raw output which *should* contain the generated queries somewhere within it
        logger.debug(f"--- Raw Output from Search Crew (Contains Generated Queries?) ---\n{final_raw_output_for_logging}\n--------------------------------------------------------------------")
        # --- END ADDED ---


        # Extract final URL list from the result object (as before)
        if search_results_object:
            raw_output = None
            # ... (rest of the existing URL extraction logic) ...
            if isinstance(search_results_object, CrewOutput):
                 raw_output = search_results_object.raw # The final output is expected here
            elif isinstance(search_results_object, str):
                 raw_output = search_results_object
            # ...

            if raw_output:
                 # Pass the FINAL raw output string to the parser
                 url_list = parse_url_list(raw_output)
                 # --- ADDED: Log the full list before slicing ---
                 logger.debug(f"Full URL list found by search execution: {url_list}")
                 # --- END ADDED ---
                 logger.info(f"Extracted {len(url_list)} URLs to potentially process: {url_list[:5]}..." if url_list else 'None')
            else:
                logger.warning("Search Crew returned output, but raw string could not be extracted for URL list.")
        else:
            logger.warning("Search Crew did not return any output object.")

    except Exception as e:
        logger.error(f"\nSearch Crew execution or parsing error: {e}", exc_info=True)

    return url_list
