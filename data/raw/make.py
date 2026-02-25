import pandas as pd

# ---------- ENRON ----------
enron = pd.read_csv("enron.csv")
enron["text"] = enron["subject"].fillna("") + "\n\n" + enron["body"].fillna("")
enron_out = enron[["text", "label"]].copy()
enron_out["source"] = "enron"
enron_out.to_csv("enron_final.csv", index=False)
print("✅ Saved: enron_final.csv")

# ---------- NAZARIO ----------
naz = pd.read_csv("nazario.csv")
naz["text"] = naz["subject"].fillna("") + "\n\n" + naz["body"].fillna("")
naz_out = naz[["text", "label"]].copy()
naz_out["source"] = "nazario"
naz_out.to_csv("nazario_final.csv", index=False)
print("✅ Saved: nazario_final.csv")

print("\nDone. Your final 'text' column is created.")
