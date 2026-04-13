import torch
import torch.nn as nn

class BuckiNetLayer(nn.Module):
    """
    Capa base de BuckiNet. 
    Aprende las potencias (Phi_p) para multiplicar las variables de entrada 
    y formar números adimensionales (Grupos Pi).
    """
    def __init__(self, num_inputs, num_pi_groups):
        super(BuckiNetLayer, self).__init__()
        # Matriz de pesos que representa los exponentes de las variables físicas
        self.Phi_p = nn.Parameter(torch.randn(num_inputs, num_pi_groups))

    def forward(self, x):
        # Advertencia: x debe contener valores estrictamente positivos (>0)
        log_x = torch.log(x)
        
        # Multiplicación matricial en el espacio logarítmico (equivale a sumar exponentes)
        pi_groups_log = torch.matmul(log_x, self.Phi_p)
        
        # Volvemos al espacio lineal (equivale a la multiplicación de las variables)
        pi_groups = torch.exp(pi_groups_log)
        
        return pi_groups


class FullBuckiNet(nn.Module):
    """
    Red completa que combina la extracción de Grupos Pi (BuckiNetLayer) 
    con un Perceptrón Multicapa (MLP) estándar para aprender la relación 
    entre los grupos adimensionales y la variable objetivo.
    """
    def __init__(self, num_inputs, num_pi_groups, num_outputs, hidden_neurons=32):
        super(FullBuckiNet, self).__init__()
        
        # 1. Capa física: descubre los grupos adimensionales
        self.pi_layer = BuckiNetLayer(num_inputs, num_pi_groups)
        
        # 2. Capa de aproximación universal: aprende la función que relaciona los grupos Pi
        self.mlp = nn.Sequential(
            nn.Linear(num_pi_groups, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, num_outputs)
        )

    def forward(self, x):
        # Extraemos los grupos Pi de las variables de entrada
        pi_groups = self.pi_layer(x)
        
        # Pasamos los grupos Pi por la red densa para obtener la predicción final
        output = self.mlp(pi_groups)
        
        # Retornamos la predicción Y la matriz de pesos Phi_p.
        # Necesitamos Phi_p fuera del modelo para aplicar la penalización del espacio nulo en la función de pérdida.
        return output, self.pi_layer.Phi_p


def buckinet_loss(predicciones, Y_train, Phi_p, D_p, lambda_null=5.0, l1_reg=1e-3, l2_reg=1e-4):
    """
    Función de pérdida informada por física.
    Combina el error estándar con restricciones dimensionales y esparsidad.
    """
    # 1. Error de reconstrucción estándar (Error Cuadrático Medio)
    mse_loss = nn.MSELoss()(predicciones, Y_train)

    # 2. Penalización del Espacio Nulo (Física)
    # Obliga a que D_p * Phi_p = 0. Es decir, que las unidades físicas se cancelen.
    null_space_penalty = torch.sum(torch.square(torch.matmul(D_p, Phi_p)))

    # 3. Regularización L1 (Esparsidad)
    # Fomenta que la red "apague" variables irrelevantes asignándoles un exponente de exactamente 0.
    l1_penalty = torch.sum(torch.abs(Phi_p))

    # 4. Regularización L2 (Control de magnitud)
    # Evita que los exponentes crezcan hacia el infinito o tomen valores absurdamente altos.
    l2_penalty = torch.sum(torch.square(Phi_p))

    # Suma ponderada de todas las pérdidas
    total_loss = mse_loss + (lambda_null * null_space_penalty) + (l1_reg * l1_penalty) + (l2_reg * l2_penalty)

    return total_loss
