class ScoreCam(object):
    """
        Esta clase ejecuta la tecnica conocida como 'ScoreScam'. La idea de esta tecnica es,
        apartir del conjunto de feature/activation maps que crea de aplicar la convolución. Los normaliza
        en el rango [0,1] para luego multiplicar esos mapa de carecteristicas con la propia imagen para
        maximizar los pixeles de la imagen original que mas han activado esa neurona, obteniendo como resultado
        un batch de N imagenes que son pasadas al modelo clasificador para devolver un nuevo array de predicciones de la clase
        que son manejadas como escores. Esos valores escalares tratados como pun... ToDo
    """
    def __init__(self, cnn_model,preprocessorFunction):
        self._cnn_model = cnn_model
        self._preprocessorFunction = preprocessorFunction
        return

    def __call__(self,
                 classifier,
                 img_array,
                 *,
                 max_N : int = -1):
        import numpy as np
        processed_img = self._preprocessorFunction(img_array.copy())
        topPred = np.argmax(classifier.predict(processed_img))
        act_map_array = self._cnn_model.predict(processed_img)

        #Controlamos que no se nos salga de los rangos permitidos:
        max_N = max_N if max_N > 0 and max_N < act_map_array.shape[3] else -1
        #Este valor sirve para realizar un Fast-ScoreCam o no(si max_N == -1 o ~= act_map_array.shape[3] )
        #Es decir, filtra el número de mapas de activación que se observan, procesando solo los mapas 
        #mas importantes.Se considera los mapas de activación más importantes  aquellos con mayor desviación tipica
        if max_N != -1: 
            #Obtenemos la desviación tipica de todos los mapas de activación
            act_map_std_list = np.array([np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])])
            #Con los siguientes metodos, ordenamos la lista anterior y cogemos el top 'max_N'
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            #Sobreescribimos el mapa de activaciones original para trabajar solo con el top 'max_N'
            act_map_array = act_map_array[:,:,:,max_N_indices]

        # 1. Normalize the raw activation value in each activation map into [0, 1]
        from ..ImageUtils import ImageUtils
        act_map_normalized_list = [ImageUtils.normalize(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]

        # 2. Resize normalized activations values
        input_shape = tuple(classifier.input.shape[1:])  # get input shape
        import cv2
        act_map_resized_list = [cv2.resize(act_map_normalized, input_shape[:2], interpolation=cv2.INTER_LINEAR) for act_map_normalized in act_map_normalized_list]

        # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
        # Aqui la parte importante, dado que esta en rango [0,1], los valores de los pixeles de imagen original solo pueden
        # mantener (si act_map_resize(i,j) == 1) o reducir (si act_map_resize(i,j) < 1). Por eso es una mascara
        masked_input_list = []
        for act_map_resized in act_map_resized_list:
            masked_input = img_array.copy().astype(np.float)
            for k in range(3):
                masked_input[0,:,:,k] *= act_map_resized
            masked_input_list.append(self._preprocessorFunction(masked_input))
        #Transform shape (N, 1, W, H, C) of masked_input_list to (N, W, H, C):
        #Lo que obtenemos es un array de N imagenes (N = channels)
        masked_input_array = np.concatenate(masked_input_list, axis=0)

        # 4. feed masked inputs into CNN model and softmax
        # ¿Que sentido tiene pasarle softmax 2 veces? (el del propio modelo y el softmax?
        # ¿A lo mejor cuando la activación de la última capa no es softmax?
        def softmax(x):
            f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
            return f
        #pred_from_masked_input_array = softmax(self.model.predict(masked_input_array))
        # Obtenemos N predicciones, una por cada 'masked_input_array' procesado anteriormente
        # Cada predicción con la probabilidad de ser una de las 'M' clases para las que la
        # red fue entrenada
        pred_from_masked_input_array = classifier.predict(masked_input_array)

        # 5. define weight as the score of target class
        # Solo nos interesa obtener las probabilidades( entendidas como puntuaciones aqui)
        # de la clase que estamos estudiando (la que devolvió el valor más alto de probabilidad 
        # en la imagen original), esta clase aqui viene referenciada pro el número guardado en 'topPred'
        scores = pred_from_masked_input_array[:,topPred]

        # 6. get final class discriminative localization map as linear weighted combination of all activation maps
        #A los mapas de activación iniciales, cada uno, lo multiplico por el escalar 'score' que le corresponde
        # Maximizando aquellos que devuelven mayor score y minimizando los que no.
        cam = np.dot(act_map_array[0,:,:,:], scores)
        cam = np.maximum(0, cam)  # Passing through ReLU
        cam /= np.max(cam)  # scale 0 to 1.0
        return ImageUtils.overlay(img_array[0],cam)#,topPred

