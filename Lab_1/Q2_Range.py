def Range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    range_Value = max(numbers) - min(numbers)
    return range_Value

if __name__ == "__main__":
    numbers = [5,3,8,1,0,4]
    results = Range(numbers)    
    print(results)