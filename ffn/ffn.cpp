//-- ## Exemplo de utilização do mlpack
//--
//-- Inclusão de dependências
//{{{
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
//}}}

int main()
{
  // Matriz que conterá os dados de treinamento e de teste carregados do CSV
  mat data;

  // Carrega e transpôe os dados
  const auto THROW_EXCEPTION = true; // lança uma exceção std::runtime_error se não conseguir carregar. default false
  const auto TRANSPOSE_INPUT = true; // transpôe a matriz depois de carregar. default true

  //-- Carrega e transpõe os dados do arquivo CSV
  //{{{
  Load( "foo.csv", data, THROW_EXCEPTION, TRANSPOSE_INPUT ); // 400 rows x 3 cols -> 3 rows x 400 cols
  cout << "Linhas:  " << data.n_rows << endl;                // 3
  cout << "Colunas: " << data.n_cols << endl;                // 400
  //}}}

  //-- Alias para os índices
  //{{{
  const auto VAR1_ROW  = 0;
  const auto VAR2_ROW  = 1;
  const auto LABEL_ROW = 2;
  const auto FIRST_COL = 0;
  const auto LAST_COL  = data.n_cols - 1;
  const auto TEST_SIZE = 10;
  //}}}

  //-- Recorta os dados de entrada para treinamento
  //{{{
  auto firtsRow  = VAR1_ROW;
  auto firtsCol  = FIRST_COL;
  auto lastRow   = VAR2_ROW;
  auto lastCol   = LAST_COL - TEST_SIZE;
  mat  traindata = data.submat( firtsRow, firtsCol, lastRow, lastCol );
  //}}}

  //-- Recorta os dados de saída para treinamento
  //{{{
  firtsRow        = LABEL_ROW;
  lastRow         = LABEL_ROW;
  mat trainlabels = data.submat( firtsRow, firtsCol, lastRow, lastCol );
  //}}}

  //-- Recorta os dados de entrada para teste
  //{{{
  firtsRow     = VAR1_ROW;
  lastRow      = VAR2_ROW;
  firtsCol     = LAST_COL - TEST_SIZE + 1;
  lastCol      = LAST_COL;
  mat testdata = data.submat( firtsRow, firtsCol, lastRow, lastCol );
  //}}}

  //-- Recorta os dados de saída para teste
  //{{{
  firtsRow       = LABEL_ROW;
  lastRow        = LABEL_ROW;
  mat testlabels = data.submat( firtsRow, firtsCol, lastRow, lastCol );
  //}}}

  //-- Dados de teste
  //{{{
  cout << "Dados de entrada para teste: \n" << testdata << endl;
  cout << "Dados de saída para teste: \n" << testlabels << endl;
  //}}}

  //-- Constrói a rede
  //{{{
  // Feed Forward Network
  // FFN< tipo de saída das camadas, regra de inicialização > model;
  FFN<MeanSquaredError<>, RandomInitialization> model;
  model.Add<Linear<>>( traindata.n_rows, 8 );
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>( 8, 8 );
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>( 8, 1 );
  model.Add<SigmoidLayer<>>();
  //}}}

  //-- Primeira sessão de treinamento
  //{{{
  for( int i = 0; i < 4; ++i ) {
    model.Train( traindata, trainlabels );
  }
  //}}}

  //-- Testa o modelo ajustado
  //{{{
  mat assignments;
  model.Predict( testdata, assignments );
  cout << "Previsões:\n" << assignments << endl;
  cout << "Classificação correta:\n" << testlabels << endl;
  //}}}

  //-- Salvando modelo para continuar depois
  //{{{
  Save( "model.xml", "model", model, false );
  //}}}

  // ...

  //-- Carregando o modelo salvo na sessão anterior
  //{{{
  Load( "model.xml", "model", model );
  //}}}

  //-- Nova sessão de treinamento para refinar
  //{{{
  for( int i = 0; i < 4; ++i ) {
    model.Train( traindata, trainlabels );
  }
  //}}}

  //-- Novos testes
  //{{{
  model.Predict( testdata, assignments );
  cout << "Previsões:\n" << assignments << endl;
  cout << "Classificação correta:\n" << testlabels << endl;
  //}}}

  //-- Salva o modelo aturalizado
  //{{{
  Save( "model2.xml", "model", model, false );
  //}}}

  return 0;
}
