import pytest

from actqa.data import Section
from actqa.pipeline import Pipeline

# A small SYNTHETIC statute used only by the tests. It is not a real Act.
_RAW = [
    ("1", "Short title", "This Act may be called the Sample Library Act."),
    ("2", "Borrowing limit", "A member may borrow at most five books at one time. A member who holds five books shall not borrow another until one is returned."),
    ("3", "Loan period", "Every book shall be returned within twenty one days of the date of issue. The librarian may extend the loan period by seven days on request."),
    ("4", "Late fee", "A member who returns a book after the due date shall pay a late fee of two rupees per day. The late fee shall not exceed the price of the book."),
    ("5", "Lost books", "A member who loses a book shall pay the full replacement cost of the book. The librarian may waive the cost for good reason shown."),
    ("6", "Membership", "Any resident above the age of twelve years may apply for membership. Membership shall be renewed every year on payment of the prescribed fee."),
    ("7", "Reference works", "Reference works shall not be taken out of the library. Reading rooms shall be open from nine in the morning to six in the evening."),
    ("8", "Digital access", "A member may access the digital catalogue through the library website. Digital loans expire automatically after fourteen days."),
    ("9", "Penalty for damage", "Whoever deliberately damages a book shall be punishable with fine which may extend to one thousand rupees. A second offence shall be punishable with fine which may extend to two thousand rupees."),
    ("10", "Appeals", "A member aggrieved by an order of the librarian may appeal to the library committee within thirty days. The committee shall decide the appeal within sixty days."),
]


@pytest.fixture(scope="session")
def mini_sections():
    return [Section("LIB", n, t, x, "Sample Library Act") for n, t, x in _RAW]


@pytest.fixture(scope="session")
def mini_pipeline(mini_sections):
    return Pipeline(mini_sections, max_chars=200)
