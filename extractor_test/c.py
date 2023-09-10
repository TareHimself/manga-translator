from extractor_test.d import another_level_of_madness


class C:
    test = 2


class D:
    f = 5


class F:
    z = 2


def example_external_method():

    arr = another_level_of_madness()

    return "THis should also be exported"
