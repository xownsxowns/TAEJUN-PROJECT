
class BankAccount:
    def __init__(self, initialBalance = 0.0):
        self._balance = initialBalance
    def deposit(self,amount):
        self._balance = self._balance + amount
    def withdraw(self, amount):
        PENALTY = 10.0
        if amount > self._balance:
            self._balance = self._balance - PENALTY
        else:
            self._balance = self._balance - amount
    def addInterest(self, rate):
        amount = self._balance * rate / 100.0
        self._balance = self._balance + amount
    def getBalance(self):
        return self._balance

harrysAccount = BankAccount(1000)
harrysAccount.deposit(500.0) # Balance is now $1500
harrysAccount.withdraw(2000.0) # Balance is now $1490
harrysAccount.addInterest(1.0) # Balance is now $1490 + 14.90
print("%.2f" % harrysAccount.getBalance())
print("Expected: 1504.90")