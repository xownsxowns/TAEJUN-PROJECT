## bank account

## A bank account has a balance that can be changed by deposits and withdrawals

class BankAccount:

    def __init__(self, initialBalance = 0.0):
        self._balance = initialBalance

    ## Deposits money into this account

    def deposit(self, amount):
        self._balance = self._balance + amount

    ## Makes a withdrawal from this account, or charges a
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

harrysAccount = BankAccount(1000.0)
harrysAccount.deposit(500.0)
harrysAccount.withdraw(2000.0)
harrysAccount.addInterest(1.0)
print("%.2f" % harrysAccount.getBalance())
print("Expected: 1504.90")