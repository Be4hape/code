import pandas as pd

# Load the Titanic dataset
data = pd.read_csv('train.csv')  # 파일 경로에 맞게 변경

# Function to process ticket data
def preprocess_ticket(ticket):
    # If the ticket contains alphabetic characters, retain only alphabetic characters
    if any(char.isalpha() for char in ticket):
        return ''.join([char for char in ticket if char.isalpha()])
    # Otherwise, return the ticket as is
    return ticket

# Apply the preprocessing function to the 'Ticket' column
data['ProcessedTicket'] = data['Ticket'].apply(preprocess_ticket)

# Rename the column 'ProcessedTicket' to 'ticketnew'
data.rename(columns={'ProcessedTicket': 'ticketnew'}, inplace=True)

# Display the processed data
print(data[['Ticket', 'ticketnew']].head())
