rate = float(input("Enter the exchange rate from EUR to KRW: "))
direct = int(input("Enter 0 to convert EUR to KRW and 1 vice versa: "))
amount = float(input("Enter the amount: "))

if direct == 0:
    result = amount * rate
    print("EUR", amount, "is KRW", result)
elif direct == 1:
    result = amount / rate
    print("KRW", amount, "is EUR", result)
