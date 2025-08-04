# import pandas as pd
# from pathlib import Path
# from newberryai import EDA
# from io import BytesIO
# import base64
# import uuid
# import matplotlib.pyplot as plt

# class EDAProcessor:
#     def __init__(self):
#         self.eda = EDA()
#         self.data = None
#         self.eda.current_data = None

#     def load_data(self, file):
#         self.data = pd.read_csv(file)
#         self.eda.current_data = self.data

#     def ask_query(self, query: str):
#         if self.eda.current_data is None:
#             return "No data loaded. Please load a CSV file first."
#         return self.eda.ask(query)

#     def visualize_all(self):
#         dist_image = self.eda.visualize_data('dist')
#         corr_image = self.eda.visualize_data('corr')
#         cat_image = self.eda.visualize_data('cat')
#         time_image = self.eda.visualize_data('time')
#         return {
#             "dist": self.convert_to_base64(dist_image),
#             "corr": self.convert_to_base64(corr_image),
#             "cat": self.convert_to_base64(cat_image),
#             "time": self.convert_to_base64(time_image)
#         }

#     def visualize_individual(self, visualization_type: str):
#         image = self.eda.visualize_data(visualization_type)
#         return {"visualization": self.convert_to_base64(image)}

#     def convert_to_base64(self, image):
#         if isinstance(image, BytesIO):
#             return "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
#         elif isinstance(image, (str, Path)) and Path(image).exists():
#             with open(image, "rb") as img_file:
#                 return "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
#         else:
#             return image if isinstance(image, str) else str(image)

# class EDAProcessorManager:
#     def __init__(self):
#         self.sessions = {}

#     def create_session(self):
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = EDAProcessor()
#         return session_id

#     def get_processor(self, session_id):
#         return self.sessions.get(session_id)



import pandas as pd
from pathlib import Path
from newberryai import EDA
from io import BytesIO
import base64
import uuid
import matplotlib.pyplot as plt

class EDAProcessor:
    def __init__(self):
        self.eda = EDA()
        self.data = None
        self.eda.current_data = None

    def load_data(self, file):
        self.data = pd.read_csv(file)
        self.eda.current_data = self.data

    def ask_query(self, query: str):
        if self.eda.current_data is None:
            return "No data loaded. Please load a CSV file first."
        return self.eda.ask(query)

    def visualize_all(self):
        return {
            "dist": self.visualize_and_get_base64('dist'),
            "corr": self.visualize_and_get_base64('corr'),
            "cat": self.visualize_and_get_base64('cat'),
            "time": self.visualize_and_get_base64('time')
        }

    def visualize_individual(self, visualization_type: str):
        return {"visualization": self.visualize_and_get_base64(visualization_type)}

    def visualize_and_get_base64(self, plot_type):
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Choose a file name per plot type, e.g., dist.png, corr.png
        file_name = f"{plot_type}.png"
        
        # Remove the old file if it exists
        if Path(file_name).exists():
            Path(file_name).unlink()

        # Try to call EDA and have it (hopefully) save the plot to file_name
        result = self.eda.visualize_data(plot_type)
        print("Returned from visualize_data:", result, type(result))

        # After the plot is generated, check if the file exists
        if Path(file_name).exists():
            with open(file_name, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode()
            # Optionally: clean up the image file after reading
            Path(file_name).unlink()
            return "data:image/png;base64," + img_b64
        else:
            return result if isinstance(result, str) else "No plot generated"



class EDAProcessorManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = EDAProcessor()
        return session_id

    def get_processor(self, session_id):
        return self.sessions.get(session_id)

