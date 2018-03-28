# Assignment3

def date_format(year, month, day, format, delimeter, abbr):
    if abbr == 1:
        if month == 1:
            month = 'Jan'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 2:
            month = 'Feb'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 3:
            month = 'Mar'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 4:
            month = 'Apr'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 5:
            month = 'May'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 6:
            month = 'Jun'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 7:
            month = 'Jul'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 8:
            month = 'Aug'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 9:
            month = 'Sep'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 10:
            month = 'Oct'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 11:
            month = 'Nov'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
        elif month == 12:
            month = 'Dec'
            if format == 'DMY':
                print(day, delimeter, month, delimeter, year, sep='')
            elif format == 'YMD':
                print(year, delimeter, month, delimeter, day, sep='')
            elif format == 'MDY':
                print(month, delimeter, day, delimeter, year, sep='')
            elif format == 'YDM':
                print(year, delimeter, day, delimeter, month, sep='')
    elif abbr == 0:
        if format == 'DMY':
            print(day, delimeter, month, delimeter, year, sep='')
        elif format == 'YMD':
            print(year, delimeter, month, delimeter, day, sep='')
        elif format == 'MDY':
            print(month, delimeter, day, delimeter, year, sep='')
        elif format == 'YDM':
            print(year, delimeter, day, delimeter, month, sep='')

year = int(input("Enter year: "))
month = int(input("Enter month: "))
day = int(input("Enter day: "))
format = input("Enter the format: ")
delimeter = input("Enter the delimeter: ")
abbr = int(input("abbr of month? (0 or 1): "))

date_format(year, month, day, format, delimeter, abbr)