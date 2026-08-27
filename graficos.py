import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar dados
df = pd.read_csv('/home/void/Documents/Estudos/classica-moveis-dados/vendas-tratado.csv') # Alterar caminho para teste
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# 1. Receita por Categoria
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análise de Vendas', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
df.groupby('CategoryName')['TotalRevenue'].sum().plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Receita Total por Categoria')
ax1.set_xlabel('Categoria')
ax1.set_ylabel('Receita (R$)')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. Quantidade de Pedidos por Modo de Envio
ax2 = axes[0, 1]
df['ShipMode'].value_counts().plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Quantidade de Pedidos por Modo de Envio')
ax2.set_xlabel('Quantidade')

# 3. Receita por Mês
ax3 = axes[1, 0]
df.groupby('OrderMonth')['TotalRevenue'].sum().plot(kind='line', ax=ax3, marker='o', color='green', linewidth=2)
ax3.set_title('Receita por Mês (Agregado)')
ax3.set_xlabel('Mês')
ax3.set_ylabel('Receita (R$)')
ax3.grid(True, alpha=0.3)

# 4. Distribuição de Valores por Prioridade
ax4 = axes[1, 1]
priority_order = ['Critical', 'High', 'Medium', 'Low']
df['OrderPriority'] = pd.Categorical(df['OrderPriority'], categories=priority_order, ordered=True)
df.boxplot(column='FinalValue', by='OrderPriority', ax=ax4)
ax4.set_title('Distribuição de Valores por Prioridade')
ax4.set_xlabel('Prioridade')
ax4.set_ylabel('Valor Final (R$)')
plt.suptitle('')

plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_1.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Top 10 Subcategorias
fig, ax = plt.subplots(figsize=(12, 6))
top_subcat = df.groupby('SubcategoryName')['TotalRevenue'].sum().nlargest(10)
top_subcat.plot(kind='barh', ax=ax, color=sns.color_palette("husl", 10))
ax.set_title('Top 10 Subcategorias por Receita', fontweight='bold', fontsize=14)
ax.set_xlabel('Receita (R$)')
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_2.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Desconto vs Receita
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['Discount %'], df['TotalRevenue'], c=df['OrderQuantity'], cmap='viridis', alpha=0.6, s=30)
ax.set_title('Correlação: Desconto vs Receita', fontweight='bold', fontsize=14)
ax.set_xlabel('Desconto (%)')
ax.set_ylabel('Receita (R$)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Quantidade')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_3.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Série Temporal de Receita
fig, ax = plt.subplots(figsize=(14, 6))
sales_by_date = df.groupby(df['OrderDate'].dt.date)['TotalRevenue'].sum()
ax.plot(sales_by_date.index, sales_by_date.values, linewidth=2, color='darkblue')
ax.fill_between(range(len(sales_by_date)), sales_by_date.values, alpha=0.3, color='lightblue')
ax.set_title('Receita Diária ao Longo do Tempo', fontweight='bold', fontsize=14)
ax.set_xlabel('Data')
ax.set_ylabel('Receita (R$)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_4.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Distribuição por Prioridade
fig, ax = plt.subplots(figsize=(10, 8))
priority_dist = df['OrderPriority'].value_counts()
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
ax.pie(priority_dist.values, labels=priority_dist.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Distribuição de Pedidos por Prioridade', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_5.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Histograma de Descontos
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df['Discount %'], bins=30, color='purple', edgecolor='black', alpha=0.7)
ax.axvline(df['Discount %'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["Discount %"].mean():.2f}')
ax.set_title('Distribuição de Descontos', fontweight='bold', fontsize=14)
ax.set_xlabel('Desconto (%)')
ax.set_ylabel('Frequência')
ax.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_6.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Vendas por Ano e Mês
fig, ax = plt.subplots(figsize=(14, 6))
sales_by_month_year = df.groupby(['OrderYear', 'OrderMonth'])['TotalRevenue'].sum().unstack()
for year in sales_by_month_year.index:
    ax.plot(sales_by_month_year.columns, sales_by_month_year.loc[year], marker='o', label=f'Ano {year}', linewidth=2)
ax.set_title('Receita Mensal por Ano', fontweight='bold', fontsize=14)
ax.set_xlabel('Mês')
ax.set_ylabel('Receita (R$)')
ax.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_7.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Top 15 Produtos
fig, ax = plt.subplots(figsize=(12, 8))
top_products = df.groupby('ProductName')['TotalRevenue'].sum().nlargest(15)
top_products.plot(kind='barh', ax=ax, color=sns.color_palette("coolwarm", 15))
ax.set_title('Top 15 Produtos por Receita', fontweight='bold', fontsize=14)
ax.set_xlabel('Receita (R$)')
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_8.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Matriz de Correlação
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = ['UnitPrice', 'OrderQuantity', 'Discount %', 'ShippingCost', 'SalesValue', 'FinalValue', 'TotalRevenue', 'DeliveryDays']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True, cbar_kws={'label': 'Correlação'})
ax.set_title('Matriz de Correlação', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_9.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Tempo de Entrega vs Receita
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['DeliveryDays'], df['TotalRevenue'], c=df['Discount %'], cmap='plasma', alpha=0.6, s=30)
ax.set_title('Tempo de Entrega vs Receita', fontweight='bold', fontsize=14)
ax.set_xlabel('Dias para Entrega')
ax.set_ylabel('Receita (R$)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Desconto (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/void/Documents/Estudos/classica-moveis-dados/graficos/grafico_10.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Todos os gráficos foram gerados com sucesso!")
print(" Gráficos salvos em: /home/void/Documents/Estudos/classica-moveis-dados/graficos/")
print("\n Arquivos gerados:")
for i in range(1, 11):
    print(f"grafico_{i}.png")