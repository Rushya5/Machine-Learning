import pandas as pd

Excel_data = {
    "City": [
        "BENGALURU", "CHENNAI", "MUMBAI", "MYSURU", "PATNA",
        "JAMMU", "GANDHI NAGAR", "HYDERABAD", "ERNAKULAM", "AMARAVATI"
    ],
    "State": ["KA", "TN", "MH", "KA", "BH", "JK", "GJ", "TS", "KL", "AP"],
    "PIN Code": [560001, 600001, 400001, 570001, 800001, 180001, 382001, 500001, 682001, 522001]
}

df = pd.DataFrame(Excel_data)
df["City, State"] = df["City"] + ", " + df["State"]
file_name = "Citys.xlsx"
df.to_excel(file_name, index=False)
loaded_df = pd.read_excel(file_name)

print("Loaded DataFrame:")
print(loaded_df)

updated_file = "Updated_City.xlsx"
loaded_df.to_excel(updated_file, index=False)

print(f"\nUpdated Excel file saved as '{updated_file}'")
