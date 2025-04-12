import threading
import cv2

def process_image(path: str):
    image = cv2.imread("image/cardiomegalia-teste.png")
    
    if image is None:
        return "Erro ao carregar a imagem."
    else:
        threading.Thread(target=(cv2.imshow("Imagem original", image))).start()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Imagem com tons de cinza", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()