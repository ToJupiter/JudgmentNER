You don’t need to change the schema—it already uses `INSERT OR REPLACE`. To resume, simply **skip rows already present** in the database when creating tasks. Here’s exactly what to add.

---

## 1. After opening the DB and creating the table, fetch completed keys

```python
    # After conn.commit() of CREATE TABLE
    cursor = conn.execute("SELECT file_name, entry_index, law_index FROM labeled_citations")
    completed = set(
        (row[0], row[1], row[2]) for row in cursor.fetchall()
    )
    print(f"Found {len(completed)} already labelled entries – will skip them")
```

---

## 2. Modify the task‑creation loop to skip completed entries

```python
    tasks = []
    for i, entry in enumerate(items):
        for j, law_text in enumerate(entry.get("laws_cited", [])):
            if (entry.get("file_name", ""), i, j) in completed:
                continue
            tasks.append(label_one(sem, i, j, law_text))
```

---

## 3. Optional: also skip when no work remains

```python
    if not tasks:
        print("All entries already labelled. Nothing to do.")
        conn.close()
        return
```

---

That’s it. Now if you stop at 20k, the next run will pick up right where you left off without sending duplicate API requests.