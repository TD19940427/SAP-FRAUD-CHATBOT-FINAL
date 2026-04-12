import pandas as pd
import numpy as np

np.random.seed(42)
# -------------------------
# LOAD DATA
# -------------------------

lfa1 = pd.read_csv(r"C:\\Users\\TANIA\\Downloads\\LFA1.csv", encoding='latin1')
lfbk = pd.read_csv(r"C:\\Users\\TANIA\\Downloads\\LFBK.csv", encoding='latin1')
rbkp = pd.read_csv(r"C:\\Users\\TANIA\\Downloads\\RBKP.csv", encoding='latin1')
bsik = pd.read_csv(r"C:\\Users\\TANIA\\Downloads\\BSIK.csv", encoding='latin1')
bsak = pd.read_csv(r"C:\\Users\\TANIA\\Downloads\\BSAK.csv", encoding='latin1')

paysim = pd.read_csv(
    r"D:\Python\AI FRAUD DETECTION PROJECT\PS_20174392719_1491204439457_log.csv", encoding='latin1')
ieee_trx = pd.read_csv(
    r"D:\Python\AI FRAUD DETECTION PROJECT\train_transaction.csv", encoding='latin1')
ieee_id = pd.read_csv(
    r"D:\Python\AI FRAUD DETECTION PROJECT\train_identity.csv", encoding='latin1')
uk = pd.read_csv(
    r"D:\Python\AI FRAUD DETECTION PROJECT\over250payments2025.csv", encoding='latin1')

print("Data Loaded Successfully")

# -------------------------
# BUILD SAP CORE
# -------------------------

bsik["status"] = "OPEN"
bsak["status"] = "CLEARED"

# List all columns that exist in BOTH rbkp and the items tables
# This prevents the creation of _x and _y suffixes
common_cols = ["BELNR", "LIFNR", "GJAHR", "BLDAT", "BUDAT"]

ap_open = rbkp.merge(bsik, on=common_cols, how="inner")
ap_cleared = rbkp.merge(bsak, on=common_cols, how="inner")

ap_sap = pd.concat([ap_open, ap_cleared], ignore_index=True)

print("Columns after concat:", ap_sap.columns.tolist())

# Join Vendor + Bank
ap_sap = ap_sap.merge(lfa1, on="LIFNR", how="left")
ap_sap = ap_sap.merge(lfbk, on="LIFNR", how="left")

# -------------------------
# SAP STANDARDIZATION
# -------------------------
# Now "BUDAT" and "LIFNR" will exist without suffixes
sap_df = ap_sap[[
    "BELNR", "LIFNR", "DMBTR", "BUDAT", "NAME1", "BANKN", "status"
]].copy()

sap_df.columns = [
    "invoice_id", "vendor_id", "amount", "date", "vendor_name", "bank_account", "status"
]

sap_df["date"] = pd.to_datetime(sap_df["date"])
sap_df["fraud_label"] = 0

print("✅ SAP data prepared")

# =========================
# 3. IEEE DATA (JOIN TRANSACTION + IDENTITY)
# =========================

ieee_full = ieee_trx.merge(ieee_id, on="TransactionID", how="left")

ieee_df = ieee_full[[
    "TransactionID",
    "TransactionAmt",
    "isFraud",
    "DeviceType"
]].copy()

ieee_df.columns = ["invoice_id", "amount", "fraud_label", "device_type"]

# Fill missing
ieee_df["device_type"] = ieee_df["device_type"].fillna("unknown")

# Create date
ieee_df["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
    np.arange(len(ieee_df)), unit="m"
)

print("✅ IEEE data prepared")

# =========================
# 4. PAYSIM DATA
# =========================

paysim_df = paysim.copy()

paysim_df["invoice_id"] = paysim_df.index
paysim_df["fraud_label"] = paysim_df["isFraud"]

paysim_df["date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
    paysim_df["step"], unit="h"
)

paysim_df = paysim_df[[
    "invoice_id", "amount", "fraud_label", "date"
]]

print("✅ PaySim data prepared")

# =========================
# 5. UK DATA (VENDOR ENRICHMENT)
# =========================

print("UK Columns found:", uk.columns.tolist())

uk_df = uk.copy()

# Map the UK specific columns to your standard names
# Based on your printout:
# 'Creditor_Name' -> vendor_name
# 'Net_Amount'    -> amount
# 'Payment_Date'  -> date

uk_df = uk_df.rename(columns={
    'Creditor_Name': 'vendor_name',
    'Net_Amount': 'amount',
    'Payment_Date': 'date'
})

# Now that they are renamed, we can safely select them
# We use errors='ignore' just in case, but the rename above should handle it
cols_to_keep = ["vendor_name", "amount", "date"]
uk_df = uk_df[cols_to_keep].dropna()

# Convert date to datetime object
uk_df["date"] = pd.to_datetime(uk_df["date"], errors='coerce')
uk_df = uk_df.dropna(subset=["date"])  # Drop rows where date conversion failed

uk_df["invoice_id"] = range(len(uk_df))
uk_df["fraud_label"] = 0

print("✅ UK data prepared")

# =========================
# 6. CREATE MASTER VENDOR LIST
# =========================

vendor_list = sap_df["vendor_id"].dropna().unique()

# Assign vendors to external datasets
ieee_df["vendor_id"] = np.random.choice(vendor_list, len(ieee_df))
paysim_df["vendor_id"] = np.random.choice(vendor_list, len(paysim_df))
uk_df["vendor_id"] = np.random.choice(vendor_list, len(uk_df))

# =========================
# 7. ALIGN DATASETS
# =========================

sap_final = sap_df[["invoice_id", "vendor_id",
                    "amount", "date", "fraud_label"]]
ieee_final = ieee_df[["invoice_id", "vendor_id",
                      "amount", "date", "fraud_label"]]
paysim_final = paysim_df[["invoice_id",
                          "vendor_id", "amount", "date", "fraud_label"]]
uk_final = uk_df[["invoice_id", "vendor_id", "amount", "date", "fraud_label"]]

# =========================
# 8. UNION ALL DATA
# =========================

final_df = pd.concat([
    sap_final,
    ieee_final,
    paysim_final,
    uk_final
], ignore_index=True)

print("✅ Data unified")

# =========================
# 9. CLEAN DATA
# =========================

final_df = final_df.dropna()
final_df = final_df[final_df["amount"] > 0]

# =========================
# 10. DUPLICATE DETECTION
# =========================

final_df["duplicate_flag"] = final_df.duplicated(
    subset=["vendor_id", "amount", "date"],
    keep=False
).astype(int)

# =========================
# 11. FEATURE ENGINEERING
# =========================

vendor_features = final_df.groupby("vendor_id").agg({
    "amount": ["count", "mean"]
}).reset_index()

vendor_features.columns = ["vendor_id", "invoice_count", "avg_amount"]

final_df = final_df.merge(vendor_features, on="vendor_id", how="left")

# Weekend flag
final_df["weekend_flag"] = final_df["date"].dt.dayofweek.isin([
                                                              5, 6]).astype(int)

# =========================
# 12. SAVE FINAL DATASET
# =========================

final_df.to_csv(
    r"D:\Python\AI FRAUD DETECTION PROJECT\final_ap_dataset.csv", index=False)

print("🎯 FINAL DATASET READY")
print(final_df.head())
