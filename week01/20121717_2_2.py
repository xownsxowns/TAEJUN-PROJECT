exchange_rate = input("Enter the exchange rate from EUR to KRW: ")
exchange_rate = float(exchange_rate)

condition = input("Enter 0 to convert EUR to KRW and 1 vice versa: ")

amount = input("Enter the amount: ")
amount = float(amount)

if condition == '0':
    print("EUR {0} is KRW {1}".format(amount, amount*exchange_rate))
elif condition == '1':
    print("KRW {0} is EUR {1}".format(amount, amount/exchange_rate))