characters_to_strip = "Ġ@"
months = {
    "JAN": "January",
    "FEB": "Febrary",
    "MAR": "March",
    "APR": "April",
    "MAY": "May",
    "JUN": "June",
    "JUL": "July",
    "AUG": "August",
    "SEP": "September",
    "OCT": "October",
    "NOV": "November",
    "DEC": "December",
}

event_prompt_sentences = (
    "The reporter witnessed the _ .",
    "This is a typical _ incident.",
)

argument_name_prompt_sentences = (
    "{} {} the _ in this {}.",
)

# all_types = {
#     "ATTACK": ["attack", "murder"],
#     "BOMBING": ["bombing", "explosion"],
#     "ARSON": ["fire"],
#     "KIDNAPPING": ["kidnapping"],
#     "ROBBERY": ["robbery"],
# }
all_event_types = {
    "ATTACK": ["attack"],
    "BOMBING": ["explosion"],
    # "ARSON": ["fire"],
    "ARSON": ["massacre"],
    "KIDNAPPING": ["kidnapping"],
    # "ROBBERY": ["robbery"],
    "ROBBERY": ["raid"],
}

interested_categories = (
    "INCIDENT: DATE", "INCIDENT: LOCATION", "INCIDENT: INSTRUMENT ID",
    "PERP: INDIVIDUAL ID", "PERP: ORGANIZATION ID",
    "PHYS TGT: ID", "HUM TGT: NAME",
)

all_arguments_types = {
    "INCIDENT: DATE": ["date"],
    "INCIDENT: LOCATION": ["location"],
    "INCIDENT: INSTRUMENT ID": ["weapon"],
    "PERP: INDIVIDUAL ID": ["suspect"],
    "PERP: ORGANIZATION ID": ["suspect"],
    "HUM TGT: NAME": ["victim"],
    "PHYS TGT: ID": ["target"],
}
