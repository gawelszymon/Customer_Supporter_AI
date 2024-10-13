
# Customer Supporter - Ogólny Plan i Zarys Projektu

## Wprowadzenie

W projekcie **Customer Supporter** celem jest stworzenie inteligentnego bota, który będzie w stanie rekomendować produkty na podstawie preferencji użytkowników.

## Zbieranie Danych

Dane do naszego modelu językowego możemy pozyskiwać za pomocą:
- **Selenium** lub **BeautifulSoup** do scrapowania danych.
- **API** sklepu, jeśli jest dostępne, do uzyskania danych do nauki naszego bota.

### Struktura Danych

Dane (kategoria, cena, specyfikacja techniczna, opinie) mogą być zarządzane i strukturyzowane w bazie danych takiej jak **LlamaIndex**. Specyficzne produkty będą oznaczane w tej strukturze za pomocą indeksów, co umożliwi LlamaIndex filtrowanie i wyciąganie odpowiednich produktów na podstawie określonych kryteriów.

## Architektura Systemu

**LangChain** zostanie użyty w celu stworzenia bota, który przeszuka naszą bazę danych LlamaIndex i wygeneruje przystępną dla użytkownika odpowiedź. LangChain wspiera integrację z modelami językowymi.

**RAG** (Retrieval-Augmented Generation) jest strukturą wyszukiwania i generowania odpowiedzi. Cały system RAG będzie odpowiedzialny za:
1. Wyszukiwanie produktów w bazie danych za pomocą LlamaIndex.
2. Generowanie rekomendacji na podstawie wyników wyszukiwania za pomocą wybranego modelu językowego, np. GPT, w zintegrowanym środowisku LangChain.
3. Generowanie odpowiedzi z wytłumaczeniem, dlaczego dany produkt jest najlepszy według wybranych preferencji.

System zbiera preferencje (dane wejściowe) na podstawie danych bezpośrednio wprowadzonych przez użytkownika. Te dane są następnie używane do filtrowania produktów w czasie rzeczywistym.

## Cykl Życia Aplikacji

Planowany cykl życia działania aplikacji obejmuje następujące kroki:
1. Użytkownik wprowadza swoje preferencje, takie jak:
   - Kategoria,
   - Maksymalna cena,
   - Marka,
   - Specyfikacja techniczna,
   - Opinie innych użytkowników.
   
2. LangChain zbiera dane wejściowe i przesyła zapytania do LlamaIndex, który wyszukuje produkty spełniające kryteria.

3. Model językowy może wygenerować odpowiedź w stylu chatBota, bądź też po prostu za pomocą GUI, gdzie wyniki będą wyświetlane jako lista opcjonalnych produktów, uporządkowana według preferencji bota.

## Technologie

Technologie, z których możemy potencjalnie skorzystać:
- **Flask** – cały backend, w tym obsługa bazy danych.
- **LangChain** – zarządzanie botem oraz jego integracja z modelem językowym.
- **LlamaIndex** – integracja z bazą danych, szybkie wyszukiwanie odpowiednich produktów na podstawie określonych preferencji.
- **Selenium/API** – gromadzenie danych o produktach.
- **SQLite/PostgreSQL** – przechowywanie informacji o produktach.
- **Vue.js/React/Svelte** – frontend, czyli tworzenie GUI lub chatBoxa oraz integracja z backendem poprzez API.