student_info = {}
keys = ('Имя', 'Возраст', 'Курс')
student_info = student_info.fromkeys(keys)

student_info['Имя'] = input('Введите имя студента: ')
student_info['Возраст'] = input('Введите возраст студента: ')
student_info['Курс'] = input('Введите на каком курсе студент: ')

for key, value in student_info.items():
    print(key + '-' + value)