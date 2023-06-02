#!/usr/bin/env python3

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import arxiv
import requests
from fitz import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv


# Defines a function to download papers from arXiv
def download(query, output_path, max_downloads, category, sort_criteria, sort_order):
    # Creates the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Prints a message indicating the search parameters
    print(
        f'[+] Searching papers with query: "{query}", max: {max_downloads}, category: {category}, sort: {sort_criteria}, order: {sort_order}'
    )

    # If a category is specified, adds it to the query
    if category is not None:
        query = f"cat:{category} AND {query}"

    # Searches for papers on arXiv using the specified parameters
    results = list(
        arxiv.Search(
            query=query,
            max_results=max_downloads,
            sort_by=sort_criteria,
            sort_order=sort_order,
        ).results()
    )

    # Prints a message indicating the number of papers to be downloaded
    print(f"[+] Downloading {len(results)} papers")

    # Defines a function to download a single paper
    def download_file(result, output_path):
        # Constructs the destination path for the downloaded file
        dest_path = os.path.join(
            output_path,
            f"{result.get_short_id()}_{result.title}.pdf".replace("/", "_").replace(
                " ", "_"
            ),
        )

        # If the file already exists, skips the download
        if os.path.exists(dest_path):
            return

        # Gets the size of the PDF file
        pdf_size = int(
            requests.request("head", result.pdf_url).headers["Content-Length"]
        )
        # Downloads the PDF file and shows a progress bar
        response = requests.get(result.pdf_url, stream=True)
        with tqdm(
                total=pdf_size,
                desc=f"{result.published.strftime('%Y-%m-%d')} {result.title[:30] + '...' if len(result.title) > 30 else result.title}",
                unit="B",
                unit_scale=True,
        ) as pbar:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Uses a thread pool to download the papers in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_file, result, output_path) for result in results
        ]

        # Waits for all downloads to complete
        for future in as_completed(futures):
            future.result()


def summarize(input_path, output_path):
    files = []
    if os.path.isdir(input_path):
        for root, _, filenames in os.walk(input_path):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    files.append(os.path.join(root, filename))
    elif os.path.isfile(input_path) and input_path.endswith(".pdf"):
        files.append(input_path)
    else:
        print(f"[-] Invalid input path: {input_path}")
        sys.exit(1)

    print(f"[+] Reading {len(files)} papers")

    os.makedirs(output_path, exist_ok=True)

    for file in files:
        document = Document(file)
        markdown = ""

        def get_page_content(page):
            return " ".join(
                [
                    str(block[4])
                    .strip()
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .replace("  ", " ")
                    .encode("utf-8", errors="ignore")
                    .decode("utf-8", errors="ignore")
                    for block in page.get_textpage().extractBLOCKS()
                    if len(
                        str(block[4])
                        .strip()
                        .replace("\n", " ")
                        .encode("utf-8", errors="ignore")
                        .decode("utf-8", errors="ignore")
                    )
                    > 0
                ]
            )

        for section in document.get_toc():
            markdown += f"## {section[1]}\n\n"
            markdown += f"{get_page_content(document[section[2] - 1])}\n\n"

        if len(markdown) < 1:
            for page in document:
                markdown += f"{get_page_content(page)}\n\n"

        print(f"[+] Extracted text from {file}")

        chunk_size = 3500  # 3500
        chunk_overlap = 400  # 400
        chunks = CharacterTextSplitter.from_tiktoken_encoder(
            separator=".", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_text(markdown)

        print(f"[+] Split text into {len(chunks)} chunks")

        chat = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL_NAME"),
            streaming=True,
            verbose=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        summaries = []

        for chunk in chunks:
            magic_number = 3500
            target_characters = round(
                (magic_number - sum([len(summary) for summary in summaries]))
                / (len(chunks) - len(summaries)),
                -1,
            )
            print(
                f"[+] Summarizing {len(summaries) + 1}/{len(chunks)} chunk (target: {target_characters} characters): {chunk[:50]}..."
            )

            response = chat(
                [
                    SystemMessage(content=os.getenv("CHUNK_SYSTEM_PROMPT")),
                    HumanMessage(
                        content=os.getenv("CHUNK_HUMAN_PROMPT")
                        .replace("{characters}", str(target_characters))
                        .replace("{chunk}", chunk)
                    ),
                ]
            )
            summaries.append(response.content)
            print()
            print()

        print(f"[+] Final summarizing...")

        response = chat(
            [
                SystemMessage(content=os.getenv("SUMMERIZE_SYSTEM_PROMPT")),
                HumanMessage(
                    content=os.getenv("SUMMERIZE_HUMAN_PROMPT").replace(
                        "{summaries}",
                        "\n\n".join(
                            [
                                f"## Summary {i + 1}\n\n{summary}"
                                for i, summary in enumerate(summaries)
                            ]
                        ),
                    )
                ),
            ]
        )

        print()
        print()

        summary = response.content

        with open(
            os.path.join(
                output_path, f"{os.path.basename(file).replace('.pdf', '.md')}"
            ),
            "w",
        ) as f:
            f.write(summary)

        print(f"[+] Summarized {file}")


# Check if the script is being run as the main program
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Check if OPENAI_API_KEY environment variable is set
    if not os.getenv("OPENAI_API_KEY"):
        print("[-] OPENAI_API_KEY is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Check if OPENAI_MODEL_NAME environment variable is set
    if not os.getenv("OPENAI_MODEL_NAME"):
        print("[-] OPENAI_MODEL_NAME is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Check if CHUNK_SYSTEM_PROMPT environment variable is set
    if not os.getenv("CHUNK_SYSTEM_PROMPT"):
        print("[-] CHUNK_SYSTEM_PROMPT is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Check if CHUNK_HUMAN_PROMPT environment variable is set
    if not os.getenv("CHUNK_HUMAN_PROMPT"):
        print("[-] CHUNK_HUMAN_PROMPT is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Check if SUMMERIZE_SYSTEM_PROMPT environment variable is set
    if not os.getenv("SUMMERIZE_SYSTEM_PROMPT"):
        print("[-] SUMMERIZE_SYSTEM_PROMPT is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Check if SUMMERIZE_HUMAN_PROMPT environment variable is set
    if not os.getenv("SUMMERIZE_HUMAN_PROMPT"):
        print("[-] SUMMERIZE_HUMAN_PROMPT is not set. Use .env file or set it manually.")
        sys.exit(1)

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(required=True, dest="handler")

    # Create a subparser for the "download" command
    parser_download = subparsers.add_parser("download")

    # Add arguments for the "download" command
    parser_download.add_argument("query", type=str, help="Arxiv search query")
    parser_download.add_argument(
        "-o",
        "--output",
        type=str,
        default="./papers",
        help="Download directory path (default: ./papers)",
    )
    parser_download.add_argument(
        "-m", "--max", type=int, default=20, help="Max download count (default: 20)"
    )
    parser_download.add_argument(
        "-c",
        "--category",
        type=str,
        default=None,
        help="Paper category (see https://arxiv.org/category_taxonomy)",
    )
    parser_download.add_argument(
        "-s",
        "--sort",
        type=arxiv.SortCriterion,
        default="relevance",
        help="Sort by (default: relevance)",
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
    )
    parser_download.add_argument(
        "-r",
        "--order",
        type=arxiv.SortOrder,
        default="descending",
        help="Sort order (default: descending)",
        choices=["ascending", "descending"],
    )

    # Set the default handler for the "download" command
    parser_download.set_defaults(handler="download")

    # Create a subparser for the "summarize" command
    parser_summarize = subparsers.add_parser("summarize")

    # Add arguments for the "summarize" command
    parser_summarize.add_argument(
        "input", type=str, help="PDF file path or directory path"
    )
    parser_summarize.add_argument(
        "-o",
        "--output",
        type=str,
        default="./summaries",
        help="Summarize output directory path (default: ./summaries)",
    )

    # Set the default handler for the "summarize" command
    parser_summarize.set_defaults(handler="summarize")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if args.handler == "download":
        download(
            args.query, args.output, args.max, args.category, args.sort, args.order
        )
    elif args.handler == "summarize":
        summarize(args.input, args.output)
    else:
        # Print help message if no command is specified
        parser.print_help()