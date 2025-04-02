# Customer Supporter - General Plan and Project Outline

## Introduction

The goal of the **Customer Supporter** project is to create an intelligent bot capable of recommending products based on user preferences.

## Data Collection

Data for our language model can be obtained through:
- **Selenium** or **BeautifulSoup** for web scraping.
- **Store API**, if available, to acquire data for training our bot.

### Data Structure

Data (category, price, technical specifications, reviews) can be managed and structured in a database such as **LlamaIndex**. Specific products will be indexed within this structure, allowing LlamaIndex to filter and retrieve the most suitable products based on defined criteria.

## System Architecture

**LangChain** will be used to develop the bot, which will search our LlamaIndex database and generate a user-friendly response. LangChain supports integration with language models.

**RAG** (Retrieval-Augmented Generation) is the framework for search and response generation. The entire RAG system will be responsible for:
1. Searching for products in the LlamaIndex database.
2. Generating recommendations based on search results using a selected language model, such as GPT, in the integrated LangChain environment.
3. Producing responses with explanations on why a particular product is the best choice based on the provided preferences.

The system collects user preferences (input data) and uses them to filter products in real-time.

## Application Lifecycle

The planned lifecycle of the application consists of the following steps:
1. The user enters their preferences, such as:
   - Category,
   - Maximum price,
   - Brand,
   - Technical specifications,
   - Reviews from other users.
   
2. LangChain gathers the input data and sends queries to LlamaIndex, which searches for products matching the criteria.

3. The language model can generate a chatbot-style response or display results in a GUI as a ranked list of recommended products.

## Technologies

Technologies we plan to use:
- **Flask** – Backend, including database management.
- **LangChain** – Managing the bot and integrating it with the language model.
- **LlamaIndex** – Database integration for fast retrieval of relevant products based on user preferences.
- **Selenium/API** – Data collection on products.
- **SQLite/PostgreSQL** – Storing product information.
- **Vue.js/React/Svelte** – Frontend development for the GUI or chatbot interface, integrating with the backend via API.
