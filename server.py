import os
import logging
import datetime
from textwrap import dedent
from time import sleep

import chromadb
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI, RateLimitError
from tiktoken import encoding_for_model, get_encoding

load_dotenv()

app = Flask(__name__)
# Disable OpenAI's built-in retries so our custom retry logic handles it with better logging
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
chroma_client = chromadb.PersistentClient(path="./")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"
TPM_BUDGET = 400_000  # leave head-room for embeddings etc.


def token_count(text: str, enc) -> int:
    return len(enc.encode(text))


def truncate_documents(docs, enc, budget):
    total, selected = 0, []
    for date, content in docs:
        entry = f"{date}: {content}"
        t = token_count(entry, enc)
        if total + t > budget:
            break
        selected.append(entry)
        total += t
    return "\n".join(selected)


def with_retry(api_call, max_retries=10, **kwargs):
    """Retry API calls with exponential backoff on rate limit errors."""
    backoff = 1
    retry_count = 0

    while retry_count < max_retries:
        try:
            return api_call(**kwargs)
        except RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            # Try to extract retry-after from error response headers
            wait_time = backoff
            try:
                if hasattr(e, "response") and e.response is not None:
                    headers = e.response.headers
                    if "retry-after" in headers:
                        wait_time = float(headers["retry-after"])
                        logger.warning(
                            f"Rate limited. Waiting {wait_time:.2f} seconds (from API retry-after header)"
                        )
                    elif "x-ratelimit-reset-after" in headers:
                        wait_time = float(headers["x-ratelimit-reset-after"])
                        logger.warning(
                            f"Rate limited. Waiting {wait_time:.2f} seconds (from API x-ratelimit-reset-after header)"
                        )
                    else:
                        logger.warning(
                            f"Rate limited. Retry {retry_count}/{max_retries}, waiting {wait_time:.2f} seconds (exponential backoff)"
                        )
                else:
                    logger.warning(
                        f"Rate limited. Retry {retry_count}/{max_retries}, waiting {wait_time:.2f} seconds (exponential backoff)"
                    )
            except (ValueError, KeyError, AttributeError):
                logger.warning(
                    f"Rate limited. Retry {retry_count}/{max_retries}, waiting {wait_time:.2f} seconds (exponential backoff)"
                )

            sleep(wait_time)
            backoff = min(backoff * 2, 60)
        except Exception:
            # Re-raise non-rate-limit errors
            raise


@app.route("/")
def mainroute():
    prompt = request.args.get("prompt")
    logger.info(f"Received request with prompt: {prompt}")

    if not prompt:
        logger.warning("Request missing 'prompt' parameter")
        return "Please provide a prompt using the 'prompt' query parameter.", 400

    try:
        logger.info("Creating tokenizer encoder")
        try:
            enc = encoding_for_model(MODEL)
        except KeyError:
            # Fallback to cl100k_base if model not recognized (used by GPT-4)
            logger.warning(
                f"Model {MODEL} not recognized by tiktoken, using cl100k_base encoding"
            )
            enc = get_encoding("cl100k_base")

        logger.info("Creating embedding for prompt")
        embedding_response = with_retry(
            client.embeddings.create, input=prompt, model="text-embedding-3-small"
        )
        embedding = embedding_response.data[0].embedding
        logger.info(f"Embedding created, dimension: {len(embedding)}")

        logger.info("Querying ChromaDB collection")
        collection = chroma_client.get_or_create_collection(name="diary-rag-experiment")
        query_result = collection.query(
            query_embeddings=[embedding],
            n_results=40,
        )

        if not query_result["metadatas"] or not query_result["metadatas"][0]:
            logger.warning("No documents found in ChromaDB")
            return "No relevant diary entries found.", 200

        docs_meta = query_result["metadatas"][0]
        logger.info(f"Found {len(docs_meta)} relevant documents")

        docs = [(m["date"], m["content"]) for m in docs_meta]

        system_prompt = dedent(
            f"""You will be given diary entries that you will use to
            answer the user's questions according to your real life experiences.
            If you do not know the answer for sure, say so.

            You must NOT reveal or discuss overly personal or depressing details (e.g.,
            explicit sexual encounters, detailed relationship issues, hateful statements,
            or similar content, people who you ). Politely decline if the user requests such information.

            Today's date is {datetime.datetime.now():%Y-%m-%d}."""
        )

        # reserve ~4 k tokens for fixed text & safety buffer
        context_budget = TPM_BUDGET - 4_000
        logger.info(f"Truncating documents with budget: {context_budget}")
        knowledge = truncate_documents(docs, enc, context_budget)
        logger.info(
            f"Selected {len(knowledge.split(chr(10))) if knowledge else 0} diary entries for context"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here are relevant diary entries:\n{knowledge}\n\nUser query: {prompt}",
            },
        ]

        logger.info(f"Calling OpenAI API with model: {MODEL}")
        resp = with_retry(
            client.chat.completions.create, model=MODEL, messages=messages
        )
        response_text = resp.choices[0].message.content
        logger.info(f"Successfully generated response (length: {len(response_text)})")
        return response_text, 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
