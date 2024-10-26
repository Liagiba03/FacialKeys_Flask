import sys
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
KEY = 'key.json'
SPREADSHEET_ID = '1R1-RdZw7W5y3h3LtZDyOoREenlU-ghKv30pCJE8XeN4'

def proceso():
    creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    
    #print("Iniciando el proceso...")
    
    # Leer datos JSON desde stdin
    data_json = sys.stdin.read()
    #print("Datos JSON recibidos:", data_json)
    
    try:
        # Convertir la cadena JSON de vuelta a un diccionario
        data = json.loads(data_json)
        
        # Extraer los valores y organizarlos en una lista
        values = list(data.values())
        values[-1] = ', '.join(map(str, values[-1]))  # Unir la última lista sin corchetes
        
        #print("Valores a insertar:", values)
        # Mandar los valores a Google Sheets
        result = sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range='A1',
            valueInputOption='USER_ENTERED',
            body={'values': [values]}  # Lista de listas, cada lista es una fila
        ).execute()

        if not values:
            print("No se insertó")
            return "No se encontraron datos."
        else:
            print("Si se insertó")
            #obtener()
            return f"Datos insertados correctamente. Celdas actualizadas: {result.get('updates').get('updatedCells')}"

    except Exception as e:
        print(f"Error al procesar los datos JSON: {str(e)}")
        return str(e)

# OBTENER EL ÚLTIMO REGISTRO Y MOSTRARLO
def obtener():
    creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()

    # Llamada a la API para obtener los valores
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range='data!A:AE').execute()
    values = result.get('values', [])

    if not values:
        return "No se encontraron datos."
    else:
        # Tomar solo el último registro
        ultimo_registro = values[-1]
        output = ', '.join(ultimo_registro)  # Convertir el último registro en una cadena de texto
        print("Si se leyó el último registro", ultimo_registro)
        return output


if __name__ == "__main__":
    result = proceso()
