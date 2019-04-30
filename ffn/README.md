[//]: <> (Documentação gerada com intmain_docmd)
## Exemplo de utilização do mlpack

Inclusão de dependências
<details>
<summary>Detalhes</summary>
<p>

```c++
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace arma;
using namespace std;

using mlpack::data::Load;
using mlpack::data::Save;
```

</p>
</details>
Carrega e transpõe os dados do arquivo CSV
<details>
<summary>Detalhes</summary>
<p>

```c++
  Load( "foo.csv", data, THROW_EXCEPTION, TRANSPOSE_INPUT ); // 400 rows x 3 cols -> 3 rows x 400 cols
  cout << "Linhas:  " << data.n_rows << endl;                // 3
  cout << "Colunas: " << data.n_cols << endl;                // 400
```

</p>
</details>
Alias para os índices
<details>
<summary>Detalhes</summary>
<p>

```c++
  const auto VAR1_ROW  = 0;
  const auto VAR2_ROW  = 1;
  const auto LABEL_ROW = 2;
  const auto FIRST_COL = 0;
  const auto LAST_COL  = data.n_cols - 1;
  const auto TEST_SIZE = 10;
```

</p>
</details>
Recorta os dados de entrada para treinamento
<details>
<summary>Detalhes</summary>
<p>

```c++
  auto firtsRow  = VAR1_ROW;
  auto firtsCol  = FIRST_COL;
  auto lastRow   = VAR2_ROW;
  auto lastCol   = LAST_COL - TEST_SIZE;
  mat  traindata = data.submat( firtsRow, firtsCol, lastRow, lastCol );
```

</p>
</details>
Recorta os dados de saída para treinamento
<details>
<summary>Detalhes</summary>
<p>

```c++
  firtsRow        = LABEL_ROW;
  lastRow         = LABEL_ROW;
  mat trainlabels = data.submat( firtsRow, firtsCol, lastRow, lastCol );
```

</p>
</details>
Recorta os dados de entrada para teste
<details>
<summary>Detalhes</summary>
<p>

```c++
  firtsRow     = VAR1_ROW;
  lastRow      = VAR2_ROW;
  firtsCol     = LAST_COL - TEST_SIZE + 1;
  lastCol      = LAST_COL;
  mat testdata = data.submat( firtsRow, firtsCol, lastRow, lastCol );
```

</p>
</details>
Recorta os dados de saída para teste
<details>
<summary>Detalhes</summary>
<p>

```c++
  firtsRow       = LABEL_ROW;
  lastRow        = LABEL_ROW;
  mat testlabels = data.submat( firtsRow, firtsCol, lastRow, lastCol );
```

</p>
</details>
Dados de teste
<details>
<summary>Detalhes</summary>
<p>

```c++
  cout << "Dados de entrada para teste: \n" << testdata << endl;
  cout << "Dados de saída para teste: \n" << testlabels << endl;
```

</p>
</details>
Constrói a rede
<details>
<summary>Detalhes</summary>
<p>

```c++
  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>( traindata.n_rows, 8 );
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>( 8, 8 );
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>( 8, 1 );
  model.Add<SigmoidLayer<>>();
```

</p>
</details>
Primeira sessão de treinamento
<details>
<summary>Detalhes</summary>
<p>

```c++
  for( int i = 0; i < 4; ++i ) {
    model.Train( traindata, trainlabels );
  }
```

</p>
</details>
Testa o modelo ajustado
<details>
<summary>Detalhes</summary>
<p>

```c++
  mat assignments;
  model.Predict( testdata, assignments );
  cout << "Previsões:\n" << assignments << endl;
  cout << "Classificação correta:\n" << testlabels << endl;
```

</p>
</details>
Salvando modelo para continuar depois
<details>
<summary>Detalhes</summary>
<p>

```c++
  Save( "model.xml", "model", model, false );
```

</p>
</details>
Carregando o modelo salvo na sessão anterior
<details>
<summary>Detalhes</summary>
<p>

```c++
  Load( "model.xml", "model", model );
```

</p>
</details>
Nova sessão de treinamento para refinar
<details>
<summary>Detalhes</summary>
<p>

```c++
  for( int i = 0; i < 4; ++i ) {
    model.Train( traindata, trainlabels );
  }
```

</p>
</details>
Novos testes
<details>
<summary>Detalhes</summary>
<p>

```c++
  model.Predict( testdata, assignments );
  cout << "Previsões:\n" << assignments << endl;
  cout << "Classificação correta:\n" << testlabels << endl;
```

</p>
</details>
Salva o modelo aturalizado
<details>
<summary>Detalhes</summary>
<p>

```c++
  Save( "model2.xml", "model", model, false );
```

</p>
</details>
