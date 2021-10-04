from inspect import currentframe, getframeinfo, getouterframes


def getframe():
    print(getframeinfo(getouterframes(currentframe())[0]))


getframe()
