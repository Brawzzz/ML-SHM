#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import json

import pandas as pd
import pickle as pkl


#============================================================================================================================#
#--------------------------------------------------------- FUNCTION ---------------------------------------------------------#
#============================================================================================================================#
def explore_file(file_path: str) -> dict:

    """
    Explore file content based on the file extension.
    Supported formats: (.csv, .xlsx, .pickle, .pkl, .json)
    
    Parameters
    ----------
    file_path : path to the file to explore

    Returns
    ----------
    datas : dictionnary containing: 'format', 'data' and 'metadata'
    """

    #---------------------------------------------
    if not os.path.exists(file_path):
        print(f"No such file or directory: '{file_path}'")
        return {}

    #------------------------------
    datas = {
        'format': None,
        'data': None, 
        'metadata': {}
    }

    extension       = os.path.splitext(file_path)[1].lower()
    datas['format'] = extension

    #------------------------------
    try:

        #---------------
        if extension == '.csv':

            datas['data']       = pd.read_csv(file_path)
            datas['metadata']   = {'source': 'csv'}

        #---------------
        elif extension in ['.xlsx', '.xls']:

            xls       = pd.ExcelFile(file_path)
            tab_0     = xls.sheet_names[0]

            datas['data']       = pd.read_excel(file_path, sheet_name=tab_0)
            datas['metadata']   = {'sheet_names': xls.sheet_names, 'extracted_sheet': tab_0}

        #---------------
        elif extension in ['.pickle', '.pkl']:

            with open(file_path, 'rb') as f:
                raw_data = pkl.load(f)

            #----------
            datas['data']   = raw_data
            metadata        = {'type': type(raw_data).__name__}

            #----------
            if isinstance(raw_data, pd.DataFrame):
                metadata['shape'] = raw_data.shape

            #----------
            elif isinstance(raw_data, dict):
                metadata['keys'] = list(raw_data.keys())

            #----------
            elif hasattr(raw_data, 'shape'):
                metadata['shape'] = raw_data.shape
                
            datas['metadata'] = metadata

        #--------------- 
        elif extension == '.json':

            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            #----------
            if isinstance(raw_data, (dict, list)):

                try:
                    df = pd.DataFrame(raw_data)

                    datas['data']       = df
                    datas['metadata']   = {'type': 'DataFrame', 'shape': df.shape}

                except ValueError:
            
                    datas['data']       = raw_data
                    datas['metadata']   = {'type': type(raw_data).__name__}

        #---------------
        else:
            print(f"extension not supported : '{extension}'")
            return datas

        return datas

    #------------------------------
    except Exception as e:
        print(f"Error while processing file : {e}")
        return datas

#================================================================================#
def display_data(datas : dict) -> None:

    """
    Display theb datas and the metadatas collected from explore_file() function
    
    Parameters
    ----------
    datas : dictionnart containing the datas, obtained with explore_file() function
    """

    #---------------------------------------------  
    if not datas or datas.get('format') is None:
        print("Erreur : Le résultat fourni est vide ou invalide.")
        return

    #------------------------------ 
    print("\n" + "="*60)
    print(f"EXPLORATION (Format : {datas.get('format').upper()})")
    print("="*60)

    #------------------------------ 
    print("\nMETADATAS :")
    metadata = datas.get('metadata', {})

    #---------------
    if not metadata:
        print("   (Aucune métadonnée disponible)")
    else:
        for key, value in metadata.items():
            print(f"   {str(key).capitalize():<15} : {value}")

    #------------------------------
    print("\nDATAS :")
    data = datas.get('data')

    #---------------
    if data is None:
        print("   (Aucune donnée brute extraite)")
    
    #---------------
    elif isinstance(data, pd.DataFrame):

        print(f"   Type : Pandas DataFrame | Dimensions : {data.shape[0]} lignes x {data.shape[1]} cols")
        print("   Aperçu des 5 premières lignes :")
        print("-" * 60)
        print(data.head(5))
        print("-" * 60)
        
    #---------------
    elif isinstance(data, dict):

        print(f"   Type : Dictionary | Keys : {len(data)}")

        for i, (k, v) in enumerate(data.items()):

            if i >= 10:
                print("      ...")
                break
                
            if hasattr(v, 'shape'):
                print(f"      - '{k}' \t\t: {type(v).__name__} (shape : {v.shape})")
            elif isinstance(v, list):
                print(f"      - '{k}' \t\t: list (length: {len(v)})")
            elif isinstance(v, dict):
                print(f"      - '{k}' \t\t: dict (keys: {len(v.keys())})")
            else:
                print(f"      - '{k}' \t\t: {type(v).__name__}")
                
    #---------------
    elif hasattr(data, 'shape'):

        print(f"   Type : {type(data).__name__} | Dimensions : {data.shape}")

        try:
            print(data[:min(3, len(data))])
        except Exception:
            print("   (error while display overview)")
            
    #---------------
    elif isinstance(data, list):

        print(f"   Type : List | Length : {len(data)}")

        for i, item in enumerate(data[:3]):
            extract = str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
            print(f"      [{i}] {type(item).__name__} : {extract}")
            
    #---------------
    else:

        print(f"   Type : {type(data).__name__}")

        repr_data = str(data)
        if len(repr_data) > 200:
            print(f"   Valeur : {repr_data[:200]}... (tronqué)")
        else:
            print(f"   Valeur : {repr_data}")
            
    print("="*60 + "\n")