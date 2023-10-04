import streamlit as st

def main():
    st.title("Simple Input App")

    # Get user input
    user_input = st.text_input("Enter something:")

    # Display the input
    st.write("You entered:", user_input)

if __name__ == "__main__":
    main()
