use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

#[derive(Clone, Debug)]
struct Input {
    a: f64,
    b: f64,
}

struct Sample {
    input: Input,
    r: f64,
}

impl Sample {
    fn new(a: f64, b: f64, r: f64) -> Sample {
        Sample {
            input: Input { a, b },
            r,
        }
    }
}

struct Hidden {
    size: usize,
    input_weights: Vec<Input>,
    output_weights: Vec<f64>,
}

impl Hidden {
    fn new(size: usize) -> Hidden {
        let rng = &mut rand::thread_rng();

        Hidden {
            size,
            input_weights: (0..size)
                .map(|_| Input {
                    a: rng.gen_range(-1.0..1.0),
                    b: rng.gen_range(-1.0..1.0),
                })
                .collect(),
            output_weights: (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        }
    }

    fn one(&mut self, learning_rate: f64, samples: &mut Vec<Sample>) {
        // 随机选择一个输入
        for i in 0..samples.len() {
            let input = &samples[i];
            let expected_output = input.r;

            // **前向传播**
            // 计算输入层到隐藏层的输出
            let hidden_layer_output = self.forward_propagation(&input.input);

            // 计算隐藏层到输出层的输出
            let output = self.calc_output(&hidden_layer_output);

            // **反向传播**
            // 计算输出层误差和梯度
            let output_error = expected_output - output;
            let output_delta = output_error * sigmoid_derivative(output);

            // 计算隐藏层误差和梯度
            let hidden_errors: Vec<f64> = self
                .output_weights
                .iter()
                .map(|w| w * output_delta)
                .collect();
            let hidden_deltas: Vec<f64> = hidden_layer_output
                .iter()
                .zip(hidden_errors.iter())
                .map(|(&h, &error)| error * sigmoid_derivative(h))
                .collect();

            // 更新权重
            for j in 0..self.size {
                self.input_weights[j].a += learning_rate * hidden_deltas[j] * input.input.a;
                self.input_weights[j].b += learning_rate * hidden_deltas[j] * input.input.b;
            }

            for j in 0..self.size {
                self.output_weights[j] += learning_rate * output_delta * hidden_layer_output[j];
            }
        }
    }

    fn forward_propagation(&mut self, input: &Input) -> Vec<f64> {
        let hidden_layer_input: Vec<f64> = self
            .input_weights
            .iter()
            .map(|v| input.a * v.a + input.b * v.b)
            .collect();

        let hidden_layer_output: Vec<f64> =
            hidden_layer_input.iter().map(|&x| sigmoid(x)).collect();
        hidden_layer_output
    }

    fn calc_output(&mut self, output: &Vec<f64>)->f64 {
        // 计算隐藏层到输出层的输出
        let output_layer_input: f64 = output
            .iter()
            .zip(self.output_weights.iter())
            .map(|(h, w)| h * w)
            .sum();
        let output = sigmoid(output_layer_input);
        output
    }

    fn estimate(&mut self, input: &Input) -> f64 {
        let hidden_layer_output = self.forward_propagation(input);
        self.calc_output(&hidden_layer_output)
    }
}

struct Knife {
    hidden: Hidden,
}

impl Knife {
    fn new() -> Self {
        Knife {
            hidden: Hidden::new(4),
        }
    }

    fn train(&mut self, epochs: usize, learning_rate: f64) {
        let mut samples: Vec<Sample> = vec![
            Sample::new(0.0, 0.0, 0.0),
            Sample::new(0.0, 1.0, 0.0),
            Sample::new(1.0, 0.0, 0.0),
            Sample::new(1.0, 1.0, 1.0),
        ];

        (0..epochs).for_each(|_| {
            self.hidden.one(learning_rate, &mut samples);
        })
    }

    fn estimate(&mut self, input: &Input) -> f64 {
        self.hidden.estimate(input)
    }
}

fn main() {
    let mut knife = Knife::new();
    knife.train(1000000, 0.1);

    println!(
        "estimate(1.0,1.0): {}",
        knife.estimate(&Input { a: 1.0, b: 1.0 })
    );
    println!(
        "estimate(1.0,0.0): {}",
        knife.estimate(&Input { a: 1.0, b: 0.0 })
    );
    println!(
        "estimate(1.0,0.7): {}",
        knife.estimate(&Input { a: 1.0, b: 0.7 })
    );
}

#[cfg(test)]
mod tests {
    use super::*; // 导入外部的功能

    #[test]
    fn test_add() {
        let a = sigmoid(1.3);
        let b = sigmoid_derivative(1.3);
        let c = sigmoid_derivative(a);
        println!("a: {:?}, b: {:?}, c: {:?}", a, b, c);
        assert_eq!(3, 3);
    }
}
