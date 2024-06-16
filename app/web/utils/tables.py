import io

import numpy as np
import pandas as pd

from app.neuro.roboflow_net.core import RoboflowPredictions


def generate_table(predictions: RoboflowPredictions) -> pd.DataFrame:
    defects_data = []
    for i in range(len(predictions.labels)):
        detection = predictions.detections[i]
        label = predictions.labels[i]
        coordinates = detection.xyxy[0].tolist()
        width = np.round(coordinates[2] - coordinates[0])
        height = np.round(coordinates[3] - coordinates[1])
        square = np.round(width * height)
        defects_data.append(
            {
                "Категория дефекта": label,
                "Ширина (px)": width,
                "Высота (px)": height,
                "Площадь (px^2)": square,
            }
        )
    return pd.DataFrame(defects_data)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer._save()
    processed_data = output.getvalue()
    return processed_data
