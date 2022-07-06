import random

# create a list of events

event = ["rock", "paper", "scissors"]


def get_computer_choice():

    return event[random.randint(0, 2)]


def get_user_choice() -> str:

    while True:
        try:
            print("please choose rock, paper or scissors!")
            user_choice = str(input())

            if user_choice.lower() in event:
                print("you have picked {}\n".format(user_choice))
                break
            else:
                pass
        except:
            print("Please choose a correct input")
            pass
    return user_choice


def get_winner(computer_choice: str, user_choice: str) -> int:

    computer_index = event.index(computer_choice)
    user_index = event.index(user_choice)

    win_computer = 0

    if computer_index == user_index:
        print("Draw")
        return "Draw"
    elif computer_index == 0 and user_index == 1:
        print("computer loses\n")
    elif computer_index == 1 and user_index == 2:
        print("computer loses\n")
    elif computer_index == 2 and user_index == 0:
        print("computer loses\n")
    else:
        print("computer wins")
        win_computer = 1
    return win_computer


def play():

    comp_choice = get_computer_choice()
    userChoice = get_user_choice()

    win = get_winner(comp_choice, userChoice)

    if win == 0:
        print("computer loses and player wins")
    else:
        print("computer wins.. better luck next time!")
    return None
