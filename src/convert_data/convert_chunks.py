"""Convert Lab 6 JSONL chunks into the Lab 7 JSON array contract."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def convert_chunks(source: Path) -> list[dict]:
    chunks = []
    seen = set()
    required = {"chunk_id", "document_id", "text", "source", "page", "metadata"}
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON") from error
            if not isinstance(chunk, dict) or not required.issubset(chunk):
                raise ValueError(f"Line {line_number}: missing required Lab 6 fields")
            if "id" in chunk or "source_id" in chunk:
                raise ValueError(f"Line {line_number}: conflicting Lab 7 field names")
            for key in ("chunk_id", "document_id", "text", "source"):
                if not isinstance(chunk[key], str) or not chunk[key].strip():
                    raise ValueError(f"Line {line_number}: invalid {key}")
            if chunk["chunk_id"] in seen:
                raise ValueError(f"Line {line_number}: duplicate chunk_id")
            if type(chunk["page"]) is not int or chunk["page"] < 1:
                raise ValueError(f"Line {line_number}: invalid PDF page")
            if not isinstance(chunk["metadata"], dict):
                raise ValueError(f"Line {line_number}: metadata must be an object")
            seen.add(chunk["chunk_id"])
            chunk["id"] = chunk.pop("chunk_id")
            chunk["source_id"] = chunk.pop("document_id")
            chunks.append(chunk)
    if not chunks:
        raise ValueError("Input contains no chunks")
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/chunks_B.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/chunks.json")
    args = parser.parse_args()
    chunks = convert_chunks(args.input)
    # Exclusive creation protects an existing dataset from accidental overwrite.
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n")
    print(f"Converted {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
