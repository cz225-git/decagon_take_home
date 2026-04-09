"""
MOCK_ORDERS — 1000 orders across ~600 customers.

On first run, generates the dataset and saves it to data/orders.json.
On subsequent runs, loads from that file. Mutations (e.g. approved refunds)
are written back to the file immediately so they persist across sessions.
"""

import json
import os
import random
from datetime import datetime, timedelta

ORDERS_PATH = "data/orders.json"

_TODAY = datetime(2026, 4, 9)
_rng = random.Random(42)

FIRST_NAMES = [
    "Alex", "Maria", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
    "Jamie", "Avery", "Quinn", "Harper", "Drew", "Blake", "Cameron", "Skylar",
    "Reese", "Finley", "Logan", "Peyton", "Emma", "Liam", "Olivia", "Noah",
    "Sophia", "Isabella", "James", "Ava", "William", "Charlotte", "Benjamin",
    "Amelia", "Lucas", "Mia", "Henry", "Evelyn", "Alexander", "Abigail",
    "Mason", "Emily", "Ethan", "Elizabeth", "Daniel", "Sofia", "Michael",
    "Owen", "Ella", "Sebastian", "Scarlett", "Aiden", "Victoria",
    "Matthew", "Madison", "Joseph", "Luna", "David", "Grace", "Carter",
    "Chloe", "Wyatt", "Penelope", "John", "Layla", "Luke", "Zoe",
    "Nathan", "Hannah", "Ryan", "Lily", "Gabriel", "Ellie", "Julian", "Nora",
    "Caleb", "Aria", "Isaiah", "Elena", "Andrew", "Naomi", "Eli", "Aurora",
]

LAST_NAMES = [
    "Johnson", "Garcia", "Lee", "Brown", "Davis", "Wilson", "Taylor", "Anderson",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Young", "Allen",
    "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Green", "Adams",
    "Baker", "Nelson", "Carter", "Mitchell", "Roberts", "Turner", "Phillips", "Campbell",
    "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers",
    "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper",
    "Richardson", "Cox", "Howard", "Ward", "Peterson", "Gray", "Ramirez", "James",
    "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes",
    "Ross", "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson",
    "Hughes", "Flores", "Washington", "Butler", "Simmons", "Foster", "Gonzales", "Bryant",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "email.com", "icloud.com", "hotmail.com"]

BOOK_TITLES = [
    "The Great Gatsby", "1984", "Atomic Habits", "Dune", "Foundation",
    "The Hobbit", "To Kill a Mockingbird", "Pride and Prejudice", "The Alchemist",
    "Sapiens", "Thinking, Fast and Slow", "The Power of Now", "Educated",
    "Becoming", "The Road", "Brave New World", "The Catcher in the Rye",
    "Lord of the Flies", "Animal Farm", "The Hitchhiker's Guide to the Galaxy",
    "Ender's Game", "The Martian", "Project Hail Mary", "Recursion",
    "Dark Matter", "Gone Girl", "The Girl with the Dragon Tattoo", "Big Little Lies",
    "Where the Crawdads Sing", "Normal People", "Circe", "The Song of Achilles",
    "The Name of the Wind", "A Way of Kings", "Mistborn", "The Final Empire",
    "Red Rising", "The Hunger Games", "Harry Potter and the Sorcerer's Stone",
    "The Fellowship of the Ring", "The Two Towers", "The Return of the King",
    "Meditations", "Man's Search for Meaning", "The 7 Habits of Highly Effective People",
    "Zero to One", "The Lean Startup", "Good to Great", "Deep Work",
    "Digital Minimalism",
]

BOOK_PRICES = {
    "The Great Gatsby": 12.99,      "1984": 13.99,
    "Atomic Habits": 18.99,         "Dune": 17.99,
    "Foundation": 16.99,            "The Hobbit": 14.99,
    "To Kill a Mockingbird": 13.99, "Pride and Prejudice": 10.99,
    "The Alchemist": 14.99,         "Sapiens": 21.99,
    "Thinking, Fast and Slow": 19.99, "The Power of Now": 16.99,
    "Educated": 17.99,              "Becoming": 19.99,
    "The Road": 15.99,              "Brave New World": 13.99,
    "The Catcher in the Rye": 12.99, "Lord of the Flies": 11.99,
    "Animal Farm": 10.99,           "The Hitchhiker's Guide to the Galaxy": 15.99,
    "Ender's Game": 16.99,          "The Martian": 17.99,
    "Project Hail Mary": 18.99,     "Recursion": 17.99,
    "Dark Matter": 16.99,           "Gone Girl": 15.99,
    "The Girl with the Dragon Tattoo": 17.99, "Big Little Lies": 15.99,
    "Where the Crawdads Sing": 16.99, "Normal People": 14.99,
    "Circe": 16.99,                 "The Song of Achilles": 15.99,
    "The Name of the Wind": 18.99,  "A Way of Kings": 22.99,
    "Mistborn": 17.99,              "The Final Empire": 17.99,
    "Red Rising": 16.99,            "The Hunger Games": 14.99,
    "Harry Potter and the Sorcerer's Stone": 19.99,
    "The Fellowship of the Ring": 18.99, "The Two Towers": 18.99,
    "The Return of the King": 18.99, "Meditations": 12.99,
    "Man's Search for Meaning": 13.99,
    "The 7 Habits of Highly Effective People": 17.99,
    "Zero to One": 19.99,           "The Lean Startup": 18.99,
    "Good to Great": 19.99,         "Deep Work": 18.99,
    "Digital Minimalism": 17.99,
}


def _display_date(d: datetime) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.strftime('%Y')}"


def _tracking_number() -> str:
    digits = "".join(str(_rng.randint(0, 9)) for _ in range(22))
    return f"USPS-{digits}"


def _generate_customers(n: int) -> list:
    """
    Build n customers with unique names.
    Shuffles all first×last combinations and takes the first n,
    guaranteeing uniqueness and deterministic output.
    """
    combos = [(f, la) for f in FIRST_NAMES for la in LAST_NAMES]
    _rng.shuffle(combos)
    combos = combos[:n]

    used_emails = set()
    customers = []
    for first, last in combos:
        domain = _rng.choice(EMAIL_DOMAINS)
        base = f"{first.lower()}.{last.lower()}"
        email = f"{base}@{domain}"
        if email in used_emails:
            email = f"{base}{_rng.randint(1, 99)}@{domain}"
        used_emails.add(email)

        phone = f"555-{_rng.randint(100, 999)}-{_rng.randint(1000, 9999)}"
        customers.append({"name": f"{first} {last}", "email": email, "phone": phone})

    return customers


def _make_order(order_id: str, customer: dict, order_date: datetime) -> dict:
    age = (_TODAY - order_date).days

    if age < 3:
        status = "processing"
    elif age < 10:
        status = "shipped"
    else:
        status = "delivered"

    n_items = _rng.choices([1, 2, 3], weights=[60, 30, 10])[0]
    items = _rng.sample(BOOK_TITLES, n_items)
    total = round(sum(BOOK_PRICES[i] for i in items), 2)

    order = {
        "order_id": order_id,
        "customer_name": customer["name"],
        "customer_email": customer["email"],
        "customer_phone": customer["phone"],
        "order_date": order_date.strftime("%Y-%m-%d"),
        "status": status,
        "items": items,
        "total": f"${total:.2f}",
        "tracking_number": None,
    }

    if status == "processing":
        est = order_date + timedelta(days=_rng.randint(7, 10))
        order["estimated_delivery"] = _display_date(est)
        order["eligible_for_refund"] = True
    elif status == "shipped":
        est = order_date + timedelta(days=_rng.randint(5, 8))
        order["estimated_delivery"] = _display_date(est)
        order["tracking_number"] = _tracking_number()
        order["eligible_for_refund"] = True
    else:
        delivery = order_date + timedelta(days=_rng.randint(4, 8))
        order["actual_delivery_date"] = _display_date(delivery)
        order["tracking_number"] = _tracking_number()
        order["eligible_for_refund"] = (_TODAY - delivery).days <= 30

    return order


def _generate() -> dict:
    orders = {}

    # Original 4 orders — kept verbatim for test compatibility
    orders["BK-1001"] = {
        "order_id": "BK-1001",
        "customer_name": "Alex Johnson",
        "customer_email": "alex.johnson@email.com",
        "customer_phone": "555-101-2020",
        "order_date": "2026-04-01",
        "status": "shipped",
        "items": ["The Great Gatsby", "1984"],
        "total": "$24.99",
        "estimated_delivery": "April 10, 2026",
        "tracking_number": "USPS-9400111899223456789012",
        "eligible_for_refund": True,
    }
    orders["BK-1002"] = {
        "order_id": "BK-1002",
        "customer_name": "Maria Garcia",
        "customer_email": "maria.garcia@email.com",
        "customer_phone": "555-202-3131",
        "order_date": "2026-02-20",
        "status": "delivered",
        "items": ["Atomic Habits"],
        "total": "$18.99",
        "actual_delivery_date": "February 27, 2026",
        "tracking_number": "USPS-9400111899223456789099",
        "eligible_for_refund": False,
    }
    orders["BK-1003"] = {
        "order_id": "BK-1003",
        "customer_name": "Sam Lee",
        "customer_email": "sam.lee@email.com",
        "customer_phone": "555-303-4242",
        "order_date": "2026-04-06",
        "status": "processing",
        "items": ["Dune", "Foundation"],
        "total": "$34.50",
        "estimated_delivery": "April 14, 2026",
        "tracking_number": None,
        "eligible_for_refund": True,
    }
    orders["BK-1004"] = {
        "order_id": "BK-1004",
        "customer_name": "Alex Johnson",
        "customer_email": "alex.johnson@email.com",
        "customer_phone": "555-101-2020",
        "order_date": "2026-04-05",
        "status": "processing",
        "items": ["The Hobbit"],
        "total": "$14.99",
        "estimated_delivery": "April 13, 2026",
        "tracking_number": None,
        "eligible_for_refund": True,
    }

    # Generate 598 customers, 996 orders (total with originals = 1000)
    customers = _generate_customers(598)

    counts = [_rng.choices([1, 2, 3, 4], weights=[40, 35, 15, 10])[0] for _ in customers]

    # Adjust total to exactly 996
    diff = sum(counts) - 996
    idxs = list(range(len(counts)))
    _rng.shuffle(idxs)
    for i in idxs:
        if diff == 0:
            break
        if diff > 0 and counts[i] > 1:
            counts[i] -= 1
            diff -= 1
        elif diff < 0:
            counts[i] += 1
            diff += 1

    order_num = 1005
    for customer, count in zip(customers, counts):
        for _ in range(count):
            days_ago = _rng.randint(0, 180)
            date = _TODAY - timedelta(days=days_ago)
            oid = f"BK-{order_num}"
            orders[oid] = _make_order(oid, customer, date)
            order_num += 1

    return orders


def save_orders():
    """Write the current state of MOCK_ORDERS to disk. Call after any mutation."""
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(MOCK_ORDERS, f, indent=2)


def _load_or_generate() -> dict:
    if os.path.exists(ORDERS_PATH):
        with open(ORDERS_PATH, encoding="utf-8") as f:
            return json.load(f)
    orders = _generate()
    os.makedirs("data", exist_ok=True)
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)
    print(f"Order data generated and saved to {ORDERS_PATH}.")
    return orders


MOCK_ORDERS = _load_or_generate()
