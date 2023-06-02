# Minipaper

An in-house tool that automatically downloads articles and summarizes them in GPT.

## Getting Started

### Prerequisites

- Python 3.9
- Poetry
- Git

### 1. Install dependencies

```bash
git clone https://github.com/qwertyroiro/minipaper
cd minipaper
poetry install
```

### 2. Copy config

```bash
cp .env.example .env
sed -i 's/OPENAI_API_KEY=""/OPENAI_API_KEY="Insert your OpenAI API key here!"/g' .env
# If you want to use GPT-4 and have access to it, run the following command.
# sed -i 's/OPENAI_MODEL_NAME="gpt-3.5-turbo"/OPENAI_MODEL_NAME="gpt-4"/g' .env
```

### 3. Run

```bash
poetry run python main.py ...
```

**Usages:**

```
usage: main.py [-h] {download,summarize} ...

positional arguments:
  {download,summarize}

optional arguments:
  -h, --help            show this help message and exit
```

**Download usage:**

```
usage: main.py download [-h] [-o OUTPUT] [-m MAX] [-c CATEGORY] [-s {relevance,lastUpdatedDate,submittedDate}] [-r {ascending,descending}] query

positional arguments:
  query                 Arxiv search query

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Download directory path (default: ./papers)
  -m MAX, --max MAX     Max download count (default: 20)
  -c CATEGORY, --category CATEGORY
                        Paper category (see https://arxiv.org/category_taxonomy)
  -s {relevance,lastUpdatedDate,submittedDate}, --sort {relevance,lastUpdatedDate,submittedDate}
                        Sort by (default: relevance)
  -r {ascending,descending}, --order {ascending,descending}
```

**Summarize usage:**

```
usage: main.py summarize [-h] [-o OUTPUT] input

positional arguments:
  input                 PDF file path or directory path

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Summarize output directory path (default: ./summaries)
```

## License

This project is licensed under the WTFPL License - see the [LICENSE](LICENSE) file for details.