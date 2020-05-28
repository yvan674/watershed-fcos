"""Ask.

Asks the user a yes or no question.
"""

def ask(question: str = 'Continue?') -> bool:
    """Asks the user to continue or not."""
    response = input(f'{question}[y/n] ')
    while True:
        if response == 'n':
            return False
        elif response == 'y':
            return True
        else:
            response = input('Please type [y] or [n] ')
