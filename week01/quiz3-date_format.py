def abbr_month(month):
    if month == 1:
        return "Jan"
    elif month == 2:
        return "Feb"
    elif month == 3:
        return "Mar"
    elif month == 4:
        return "Apr"
    elif month == 5:
        return "May"
    elif month == 6:
        return "Jun"
    elif month == 7:
        return "Jul"
    elif month == 8:
        return "Aug"
    elif month == 9:
        return "Sep"
    elif month == 10:
        return "Oct"
    elif month == 11:
        return "Nov"
    elif month == 12:
        return "Dec"


def date_format(year, month, day, format, delimeter, abbr):
    if format == "DMY":
        if abbr == 1:
            month = abbr_month(month)
        print(day, month, year, sep=delimeter)
    elif format == "YMD":
        if abbr == 1:
            month = abbr_month(month)
        print(year, month, day, sep=delimeter)
    elif format == "MDY":
        if abbr == 1:
            month = abbr_month(month)
        print(month, day, year, sep=delimeter)
    elif format == "YDM":
        if abbr == 1:
            month = abbr_month(month)
        print(year, day, month, sep=delimeter)


year = int(input("Enter year: "))
month = int(input("Enter month: "))
day = int(input("Enter day: "))
format = input("Enter the format: ")
delimeter = input("Enter the delimeter: ")
abbr = int(input("abbr of month? (0 or 1): "))

date_format(year, month, day, format, delimeter, abbr)
