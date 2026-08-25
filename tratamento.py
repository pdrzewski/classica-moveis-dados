import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataTreatment:
    """Classe para tratamento e limpeza de dados de vendas"""
    
    def __init__(self, file_path):
        """Inicializar com caminho do arquivo"""
        self.file_path = file_path
        self.df = None
        self.df_original = None
        logger.info(f"Iniciando tratamento do arquivo: {file_path}")
    
    def load_data(self):
        """Carregar dados do CSV"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.df_original = self.df.copy()
            logger.info(f"✓ Arquivo carregado com sucesso: {self.df.shape[0]} linhas e {self.df.shape[1]} colunas")
            return self
        except Exception as e:
            logger.error(f"✗ Erro ao carregar arquivo: {e}")
            raise
    
    def display_info(self):
        """Exibir informações iniciais dos dados"""
        logger.info("\n=== INFORMAÇÕES INICIAIS ===")
        logger.info(f"Dimensões: {self.df.shape}")
        logger.info(f"\nTipos de dados:\n{self.df.dtypes}")
        logger.info(f"\nValores faltantes:\n{self.df.isnull().sum()}")
        return self
    
    def convert_dates(self):
        """Converter colunas de data para formato datetime"""
        try:
            logger.info("\n=== CONVERTENDO DATAS ===")
            date_columns = ['OrderDate', 'DeliveryDate']
            
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col])
                    logger.info(f"✓ {col} convertida para datetime")
        except Exception as e:
            logger.error(f"✗ Erro ao converter datas: {e}")
        
        return self
    
    def convert_types(self):
        """Converter tipos de dados apropriados"""
        try:
            logger.info("\n=== CONVERTENDO TIPOS DE DADOS ===")
            
            # Converter numéricas
            numeric_columns = ['CustomerKey', 'ProductKey', 'UnitPrice', 
                             'OrderQuantity', 'Discount %', 'ShippingCost']
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    logger.info(f"✓ {col} convertida para numérico")
            
            # Converter categóricas
            categorical_columns = ['ShipMode', 'CategoryName', 'SubcategoryName', 'OrderPriority']
            
            for col in categorical_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
                    logger.info(f"✓ {col} convertida para categoria")
            
        except Exception as e:
            logger.error(f"✗ Erro ao converter tipos: {e}")
        
        return self
    
    def handle_missing_values(self):
        """Lidar com valores faltantes"""
        try:
            logger.info("\n=== TRATANDO VALORES FALTANTES ===")
            
            missing_counts = self.df.isnull().sum()
            if missing_counts.sum() > 0:
                logger.info("Valores faltantes encontrados:")
                logger.info(missing_counts[missing_counts > 0])
                
                # Preencher valores faltantes em colunas numéricas com a mediana
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if self.df[col].isnull().sum() > 0:
                        median_value = self.df[col].median()
                        self.df[col].fillna(median_value, inplace=True)
                        logger.info(f"✓ {col}: preenchidos com mediana ({median_value})")
                
                logger.info("✓ Valores faltantes tratados")
            else:
                logger.info("✓ Nenhum valor faltante encontrado")
        
        except Exception as e:
            logger.error(f"✗ Erro ao tratar valores faltantes: {e}")
        
        return self
    
    def remove_duplicates(self):
        """Remover registros duplicados"""
        try:
            logger.info("\n=== REMOVENDO DUPLICATAS ===")
            
            initial_rows = len(self.df)
            
            # Remover duplicatas completas
            self.df.drop_duplicates(inplace=True)
            
            removed = initial_rows - len(self.df)
            if removed > 0:
                logger.info(f"✓ {removed} registros duplicados removidos")
            else:
                logger.info("✓ Nenhuma duplicata encontrada")
        
        except Exception as e:
            logger.error(f"✗ Erro ao remover duplicatas: {e}")
        
        return self
    
    def create_derived_columns(self):
        """Criar colunas derivadas úteis"""
        try:
            logger.info("\n=== CRIANDO COLUNAS DERIVADAS ===")
            
            # Valor total da venda (sem desconto)
            if 'UnitPrice' in self.df.columns and 'OrderQuantity' in self.df.columns:
                self.df['SalesValue'] = self.df['UnitPrice'] * self.df['OrderQuantity']
                logger.info("✓ SalesValue criado (UnitPrice * OrderQuantity)")
            
            # Valor do desconto
            if 'Discount %' in self.df.columns:
                self.df['DiscountValue'] = self.df['SalesValue'] * (self.df['Discount %'] / 100)
                logger.info("✓ DiscountValue criado")
            
            # Valor final (após desconto)
            self.df['FinalValue'] = self.df['SalesValue'] - self.df['DiscountValue']
            logger.info("✓ FinalValue criado (SalesValue - DiscountValue)")
            
            # Lucro estimado (FinalValue + ShippingCost)
            if 'ShippingCost' in self.df.columns:
                self.df['TotalRevenue'] = self.df['FinalValue'] + self.df['ShippingCost']
                logger.info("✓ TotalRevenue criado")
            
            # Dias para entrega
            if 'OrderDate' in self.df.columns and 'DeliveryDate' in self.df.columns:
                self.df['DeliveryDays'] = (self.df['DeliveryDate'] - self.df['OrderDate']).dt.days
                logger.info("✓ DeliveryDays criado")
            
            # Mês e Ano do pedido
            if 'OrderDate' in self.df.columns:
                self.df['OrderMonth'] = self.df['OrderDate'].dt.month
                self.df['OrderYear'] = self.df['OrderDate'].dt.year
                self.df['OrderYearMonth'] = self.df['OrderDate'].dt.strftime('%Y-%m')
                logger.info("✓ OrderMonth, OrderYear e OrderYearMonth criados")
        
        except Exception as e:
            logger.error(f"✗ Erro ao criar colunas derivadas: {e}")
        
        return self
    
    def validate_data(self):
        """Validar dados tratados"""
        try:
            logger.info("\n=== VALIDANDO DADOS ===")
            
            # Verificar valores negativos em colunas que não devem ter
            positive_columns = ['UnitPrice', 'OrderQuantity', 'ShippingCost', 
                              'SalesValue', 'DiscountValue', 'FinalValue', 'TotalRevenue']
            
            for col in positive_columns:
                if col in self.df.columns:
                    negative_count = (self.df[col] < 0).sum()
                    if negative_count > 0:
                        logger.warning(f"⚠ {col}: {negative_count} valores negativos encontrados")
                    else:
                        logger.info(f"✓ {col}: sem valores negativos")
            
            # Verificar valores de desconto válidos (0-100%)
            if 'Discount %' in self.df.columns:
                invalid_discount = ((self.df['Discount %'] < 0) | (self.df['Discount %'] > 1)).sum()
                if invalid_discount > 0:
                    logger.warning(f"⚠ Discount %: {invalid_discount} valores inválidos (fora de 0-1)")
                else:
                    logger.info("✓ Discount %: valores válidos")
            
            # Verificar se delivery data >= order date
            if 'OrderDate' in self.df.columns and 'DeliveryDate' in self.df.columns:
                invalid_dates = (self.df['DeliveryDate'] < self.df['OrderDate']).sum()
                if invalid_dates > 0:
                    logger.warning(f"⚠ {invalid_dates} registros com DeliveryDate < OrderDate")
                else:
                    logger.info("✓ Todas as datas de entrega são posteriores aos pedidos")
        
        except Exception as e:
            logger.error(f"✗ Erro ao validar dados: {e}")
        
        return self
    
    def display_summary(self):
        """Exibir resumo dos dados tratados"""
        logger.info("\n=== RESUMO FINAL ===")
        logger.info(f"Linhas: {self.df.shape[0]} | Colunas: {self.df.shape[1]}")
        logger.info(f"\nColunas do dataset:\n{list(self.df.columns)}")
        logger.info(f"\nPrimeiras linhas:\n{self.df.head()}")
        return self
    
    def save_treated_data(self, output_path=None):
        """Salvar dados tratados em novo CSV"""
        try:
            if output_path is None:
                output_path = '/home/void/Documents/Estudos/classica-moveis-dados/Office Sales-Sales.csv'
            
            logger.info(f"\n=== SALVANDO DADOS ===")
            self.df.to_csv(output_path, index=False)
            logger.info(f"✓ Dados tratados salvos em: {output_path}")
            
            # Salvar também em Excel para melhor visualização
            excel_path = output_path.replace('.csv', '.xlsx')
            self.df.to_excel(excel_path, index=False, sheet_name='Sales')
            logger.info(f"✓ Dados também salvos em: {excel_path}")
            
        except Exception as e:
            logger.error(f"✗ Erro ao salvar dados: {e}")
        
        return self
    
    def generate_report(self):
        """Gerar relatório estatístico dos dados"""
        try:
            logger.info("\n=== RELATÓRIO ESTATÍSTICO ===")
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            stats = self.df[numeric_cols].describe()
            
            logger.info(f"\nEstatísticas das colunas numéricas:\n{stats}")
            
            # Estatísticas por categoria
            if 'OrderPriority' in self.df.columns:
                logger.info(f"\nDistribuição por Prioridade:\n{self.df['OrderPriority'].value_counts()}")
            
            if 'ShipMode' in self.df.columns:
                logger.info(f"\nDistribuição por Modo de Envio:\n{self.df['ShipMode'].value_counts()}")
            
            if 'CategoryName' in self.df.columns:
                logger.info(f"\nDistribuição por Categoria:\n{self.df['CategoryName'].value_counts()}")
        
        except Exception as e:
            logger.error(f"✗ Erro ao gerar relatório: {e}")
        
        return self
    
    def execute_treatment(self, output_path=None):
        """Executar o tratamento completo dos dados"""
        return (self
                .load_data()
                .display_info()
                .convert_dates()
                .convert_types()
                .handle_missing_values()
                .remove_duplicates()
                .create_derived_columns()
                .validate_data()
                .display_summary()
                .generate_report()
                .save_treated_data(output_path))


# Script de execução
if __name__ == "__main__":
    # Caminho do arquivo
    input_file = '/home/void/Documents/Estudos/classica-moveis-dados/Office Sales-Sales.csv'
    output_file = '/home/void/Documents/Estudos/classica-moveis-dados/vendas-tratado.csv'
    
    # Executar tratamento
    treatment = DataTreatment(input_file)
    treatment.execute_treatment(output_file)
    
    logger.info("\n" + "="*60)
    logger.info("TRATAMENTO DE DADOS CONCLUÍDO COM SUCESSO!")
    logger.info("="*60)