import streamlit as st
import pandas as pd
from apriori import apriori
from association_rules import association_rules
from utils import TransactionEncoder, get_transactions
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Apriori - Frequent Itemset Mining",
    page_icon=":rocket:",
    # layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    
    data_frame = None
    
    st.title("APRIORI - Frequent Itemset Mining")
    
    st.subheader(":blue[Choix du type de fichier.]")
    
    file_type = st.selectbox(
        "Sélectionnez type de fichier",
        ["Télécharger un fichier csv","Télécharger un fichier txt"]
    )
    
    if file_type == "Télécharger un fichier csv":
        
        st.subheader(":blue[Importation de notre data set.]")
        file = st.file_uploader("Importer le fichier", type="csv")
        
        if file is not None:
            data_frame = pd.read_csv(file, on_bad_lines='skip')  
        
    elif file_type == "Télécharger un fichier txt":
        
        st.subheader(":blue[Importation de notre data set.]")
        file = st.file_uploader("Importer le fichier", type="txt")
        
        
        st.subheader(":blue[Séparateur du fichier txt.]")
        separator_label = st.selectbox(
            "Sélectionnez un séparator",
            ["Espace","Virgule", "Point Virgule"]
        )
        
        
        if separator_label=="Espace":
            separator = " "
        elif separator_label == "Virgule":
            separator =  ","
        elif separator_label =="Point Virgule":
            separator = ";"

        if file is not None:
            file_lines = file.readlines()
            
            data = []
            for line in file_lines:
                array = line.decode('utf-8').strip().split(separator)
                data.append(array)
            
            data_frame = pd.DataFrame(data=data)
    
    execution_times = []

    start_time = time.time()
            
    if data_frame is not None:
        
        st.subheader(":blue[Aperçu de notre dataset.]")

        st.dataframe(data_frame.head())
        
        
        st.subheader(":blue[Support mininum.]")

        min_support = st.number_input(
            'ENTREZ LE SUPPORT MINIMUM: ' ,
            step=1, 
            format="%d",
            min_value=1,
            max_value=100
        )
        
        st.subheader(":blue[Niveau de confiance mininum.]")
        
        min_confidence = st.number_input(
            'ENTREZ LE NIVEAU DE CONFIANCE MINIMUM: ' ,
            step=1, 
            format="%d",
            value=50,
            min_value=20,
            max_value=100
        )
                
        transactions = get_transactions(data_frame)
        
        te = TransactionEncoder()
        df = te.fit_transform(transactions, set_pandas=True)
        
        freq_items = apriori(df, (min_support/100))
        
        freq_items_df = pd.DataFrame(freq_items)
        
        freq_items_df['itemsets'] = freq_items_df['itemsets'].apply(lambda x: ', '.join(x))
        
        st.subheader(":blue[Résultats frequent items en fonction du support défini.]")
        st.dataframe(freq_items_df, width=1500)
        
        st.subheader(":blue[Top 10 des recommandations les plus bénéfiques.]")

        if len(freq_items) > 0:
            rules = association_rules(freq_items, metric="confidence", min_threshold=(min_confidence/100))
            
            top_10_rules  = rules.sort_values('confidence')[:10]
                    
            top_10_rules["consequents"] = top_10_rules["consequents"].apply(lambda x: ', '.join(x))
            top_10_rules['antecedents'] = top_10_rules['antecedents'].apply(lambda x: ', '.join(x))

            st.dataframe(top_10_rules, width=2000)
        
        st.subheader(":blue[Courbe montrant lʼévolution du temps dʼexécution en fonction du support]")
        option = st.selectbox('Voulez-vous afficher le temps d\'exécution de l\'algo en fonction du support', ["Non", "Oui"])
        if option == "Oui":
            support_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
            execution_times = []

            for sup in support_values:
                start_time = time.time()
                
                apriori(df, (min_support/100))
                rules = association_rules(freq_items, metric="confidence", min_threshold=(min_confidence/100))
              
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
            
            fig, ax = plt.subplots()
            plt.plot(support_values, execution_times, marker='o')
            plt.xlabel('Support (%)')
            plt.ylabel('Temps d\'exécution (s)')
            plt.title('Évolution du temps d\'exécution en fonction du support')
            plt.grid(True)
            st.pyplot(fig)

if __name__ == "__main__":
    main()