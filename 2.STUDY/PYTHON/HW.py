### Problem 1
import turtle
import numpy as np

def distance(x1, y1, x2, y2):
    dist = np.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return dist

x1 = int(turtle.textinput("", "x1"))
x2 = int(turtle.textinput("", "x2"))
y1 = int(turtle.textinput("", "y1"))
y2 = int(turtle.textinput("", "y2"))
xy1 = (x1, y1)
xy2 = (x2, y2)
distance = distance(x1, y1, x2, y2)

turtle.pensize(10)
turtle.color('blue')
turtle.penup()
turtle.goto(xy1)
turtle.pendown()
turtle.goto(xy2)

turtle.hideturtle()
turtle.write(str(distance), font=("Arial", 16, "normal"))

turtle.exitonclick()


### Problem 2
# 4. 사용자로부터 5개의 도시 이름을 입력받아 citylist라는 이름의 리스트에 저장하고, 가장 마지막에 입력한 도시부터 가장 처음에 입력한 도시까지의 도시 이름을 출력하는 프로그램을 작성하시오.
#    조건 1) 공백 리스트를 먼저 생성할 것
#    조건 2) 리스트의 이름은 citylist로 지정할 것
#    조건 3) 띄어쓰기(blank or space)와 콤마의 표현에 주의하여 결과 화면과 같이 나타날 수 있도록 할 것

citylist = list()
order_list = ['첫번째', '두번째', '세번째', '네번째', '다섯번째']
for i in range(5):
    city_name = input(order_list[i]+' 도시 이름을 입력하세요 :')
    citylist.append(city_name)

citylist.reverse()
print('입력한 도시 이름 = ', ', '.join(citylist))
