# prepare_rag_data.py

import pandas as pd
import re
from bs4 import BeautifulSoup
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def clean_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text()


def anonymize_text(text, analyzer, anonymizer):
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def preprocess_and_anonymize(input_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Basic cleanup
    df = df.dropna(subset=["thread_content"])
    df["sent_date_and_time"] = pd.to_datetime(df["sent_date_and_time"], errors="coerce")
    df["clean_content"] = df["thread_content"].apply(clean_html)

    # Set up Presidio
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    df["anonymized_content"] = df["clean_content"].apply(lambda text: anonymize_text(text, analyzer, anonymizer))
    return df[["ticket_id", "sent_date_and_time", "anonymized_content"]].dropna()


def group_threads(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    df["sent_date_and_time"] = pd.to_datetime(df["sent_date_and_time"], errors="coerce")
    df = df.dropna(subset=["ticket_id", "anonymized_content"])

    grouped = df.groupby("ticket_id")
    csv_rows = []
    jsonl_rows = []

    for ticket_id, group in grouped:
        messages = group.sort_values("sent_date_and_time")["anonymized_content"].dropna().astype(str).tolist()
        full_thread = "\n\n---\n\n".join(messages)
        start = group["sent_date_and_time"].min()
        end = group["sent_date_and_time"].max()

        csv_rows.append({
            "ticketId": ticket_id,
            "createdTime_start": start,
            "createdTime_end": end,
            "content": full_thread
        })
        jsonl_rows.append({
            "id": str(ticket_id),
            "text": full_thread
        })

    return csv_rows, jsonl_rows


def save_rag_ready_data(csv_rows: list[dict], jsonl_rows: list[dict], csv_path: str, jsonl_path: str):
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    with open(jsonl_path, "w") as f:
        for row in jsonl_rows:
            f.write(f"{row}\n")


def run_pipeline(input_csv: str, csv_output: str, jsonl_output: str):
    df_clean = preprocess_and_anonymize(input_csv)
    csv_rows, jsonl_rows = group_threads(df_clean)
    save_rag_ready_data(csv_rows, jsonl_rows, csv_output, jsonl_output)


if __name__ == "__main__":
    run_pipeline(
        input_csv="zoho_current_tickets_03-18-2025.csv",
        csv_output="data/rag_ready_threads.csv",
        jsonl_output="data/rag_ready_threads.jsonl"
    )
