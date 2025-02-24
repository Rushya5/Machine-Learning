date = "01-JUN-2021"

day,month,year= date.split('-')
month_Map = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

month_Number = month_Map[month.upper()]

print("Day:", day)
print("Month:", month)
print("Numerical Month:", month_Number)
print("Year:", year)