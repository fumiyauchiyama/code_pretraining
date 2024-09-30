import ast
import random
import string
from typing import Set, List
import keyword
from datasets import load_dataset

try:
    # Python 3.9以降
    from ast import unparse
except ImportError:
    # Python 3.8以前ではastunparseを使用
    import astunparse

def collect_variable_names(node: ast.AST) -> Set[str]:
    """
    ASTノードを走査して、プログラム内で使用されているすべての変数名を収集します。

    Args:
        node (ast.AST): ASTのルートノード。

    Returns:
        Set[str]: 収集された変数名のセット。
    """
    variable_names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef):
            variable_names.add(child.name)
            for arg in child.args.args:
                variable_names.add(arg.arg)
        elif isinstance(child, ast.ClassDef):
            variable_names.add(child.name)
        elif isinstance(child, ast.Name):
            variable_names.add(child.id)
        elif isinstance(child, ast.arg):
            variable_names.add(child.arg)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                variable_names.add(alias.name)
                if alias.asname:
                    variable_names.add(alias.asname)                    
        elif isinstance(child, ast.ImportFrom):
            variable_names.add(child.module)
            for alias in child.names:
                variable_names.add(alias.name)
                if alias.asname:
                    variable_names.add(alias.asname)
        elif isinstance(child, ast.Attribute):
            variable_names.add(child.attr)  # 属性名を追加
        elif isinstance(child, ast.Global):
            for name in child.names:
                variable_names.add(name)
    # 除外リスト: キーワードや組み込み関数名
    builtins = set(dir(__builtins__))
    keywords = set(keyword.kwlist)
    variable_names = variable_names - builtins - keywords

    return variable_names

class VariableScrambler(ast.NodeTransformer):
    def __init__(
            self, 
            variable_names: List[str], 
            scrambled_name: bool = True,
            keep_dependencies: bool = False,
            length: int = 8,
            ):
        super().__init__()
        self.variable_names = variable_names
        self.scrambled_name = scrambled_name
        self.length = length
        self.keep_dependencies = keep_dependencies
        self.identifier_map = {}
        if keep_dependencies:
            for name in variable_names:
                first_char = random.choice(string.ascii_letters + '_')
                other_chars = ''.join(random.choices(string.ascii_letters + string.digits + '_', k=self.length-1))
                self.identifier_map[name] = first_char + other_chars

    def get_random_name(self, id: str) -> str:
        """既存の変数名からランダムに選択します。"""
        if self.scrambled_name:
            return random.choice(self.variable_names)
        else:
            if self.keep_dependencies:
                builtins = set(dir(__builtins__))
                keywords = set(keyword.kwlist)

                if id in builtins or id in keywords:
                    return id
                else:
                    return self.identifier_map[id]
            
            first_char = random.choice(string.ascii_letters + '_')
            other_chars = ''.join(random.choices(string.ascii_letters + string.digits + '_', k=self.length-1))
            return first_char + other_chars

    def visit_Name(self, node: ast.Name):
        # 変数の使用箇所をランダムな既存の変数名に変更
        node.id = self.get_random_name(node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # 関数名をランダムな既存の変数名に変更
        node.name = self.get_random_name(node.name)
        # 引数名もランダムに変更
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        # クラス名をランダムな既存の変数名に変更
        node.name = self.get_random_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg):
        # 引数名をランダムな既存の変数名に変更
        node.arg = self.get_random_name(node.arg)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # 属性名をランダムな既存の変数名に変更
        node.attr = self.get_random_name(node.attr)
        self.generic_visit(node)
        return node

    def visit_Global(self, node: ast.Global):
        # グローバル変数名をランダムな既存の変数名に変更
        node.names = [self.get_random_name(name) for name in node.names]
        return node

    def visit_Import(self, node: ast.Import):
        # インポート名をランダムな既存の変数名に変更
        for alias in node.names:
            alias.name = self.get_random_name(alias.name)
            if alias.asname:
                alias.asname = self.get_random_name(alias.asname)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # from ... import ... のインポート名をランダムな既存の変数名に変更
        node.module = self.get_random_name(node.module)
        for alias in node.names:
            alias.name = self.get_random_name(alias.name)
            if alias.asname:
                alias.asname = self.get_random_name(alias.asname)
        return node

def scramble_variable_names(source_code: str) -> str:
    """
    Pythonソースコード内の変数名をプログラム内の既存の変数名にランダムにシャッフルして入れ替えます。
    変数名の依存関係や参照を無視するため、生成されたコードは意味が破壊されます。

    Args:
        source_code (str): 入力のPythonソースコード。

    Returns:
        str: 変数名がランダムにシャッフルされたPythonソースコード。
    """
    try:
        tree = ast.parse(source_code)
    except:
        return ""
    
    try:
        variable_names = list(collect_variable_names(tree))
        if not variable_names:
            return source_code  # 変数名がない場合は元のコードを返す

        scrambler = VariableScrambler(variable_names)
        new_tree = scrambler.visit(tree)
        ast.fix_missing_locations(new_tree)
        if 'unparse' in globals():
            new_code = unparse(new_tree)
        else:
            new_code = astunparse.unparse(new_tree)
    except Exception as e:
        print("Error on scramble")
        raise e
        return ""

    return new_code

def randomize_variable_names(source_code: str, length=8, keep_dependencies=False) -> str:
    """
    Pythonソースコード内の変数名をランダムな文字列に入れ替えます。
    変数名の依存関係や参照を無視するため、生成されたコードは意味が破壊されます。

    Args:
        source_code (str): 入力のPythonソースコード。

    Returns:
        str: 変数名がランダムにシャッフルされたPythonソースコード。
    """
    try:
        tree = ast.parse(source_code)
    except:
        return ""
    
    try:
        variable_names = list(collect_variable_names(tree))
        if not variable_names:
            return source_code  # 変数名がない場合は元のコードを返す

        scrambler = VariableScrambler(
            variable_names, 
            scrambled_name=False, 
            length=length,
            keep_dependencies=keep_dependencies,
            )
        new_tree = scrambler.visit(tree)
        ast.fix_missing_locations(new_tree)
        if 'unparse' in globals():
            new_code = unparse(new_tree)
        else:
            new_code = astunparse.unparse(new_tree)
    except Exception as e:
        print("Error on randomize_variable_names")
        raise e
        #return ""

    return new_code

# コメントの削除
class RemoveStringExpressions(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # このノードを削除
        return node

def remove_string_comments(source_code):
    # ASTにパース
    try:
        tree = ast.parse(source_code)
    except:
        return ""
    
    # 文字列定数を削除
    transformer = RemoveStringExpressions()
    try:
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        if 'unparse' in globals():
            new_code = unparse(new_tree)
        else:
            new_code = astunparse.unparse(new_tree)
    except:
        print("Error on remove_string_comments")
        return ""

    return new_code

def make_ablation_dataset(batch):
    output = {
        'content': [i for i in batch['content']],
        'content_no_comment': [remove_string_comments(i) for i in batch['content']],
        'content_scrambled': [scramble_variable_names(i) for i in batch['content']],
    }
    output['content_no_comment_scrambled'] = [scramble_variable_names(i) for i in output['content_no_comment']]
    return output

def add_randomized_code(batch):
    output = {
        'content_randomized': [randomize_variable_names(i) for i in batch['content']],
        'content_no_comment_randomized': [randomize_variable_names(i) for i in batch['content_no_comment']],
    }
    return output

def add_randomized_code_with_dependencies(batch):
    output = {
        'content_randomized_w_dep': [randomize_variable_names(i, keep_dependencies=True) for i in batch['content']],
        'content_no_comment_randomized_w_dep': [randomize_variable_names(i, keep_dependencies=True) for i in batch['content_no_comment']],
    }
    return output

if __name__ == '__main__':
    # データセットの読み込み
    # dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:500000]", verification_mode="no_checks")
    # dataset = dataset.map(make_ablation_dataset, batched=True, num_proc=10, remove_columns=dataset.column_names)
    # dataset = dataset.filter(lambda x: x['content_no_comment'] != "" and x['content_scrambled'] != "" and x['content_no_comment_scrambled'] != "")
    dataset = load_dataset("fumiyau/python_ablation_500000", split="train",)

    num_samples_before = len(dataset)
    print(num_samples_before)

    dataset = dataset.map(make_ablation_dataset, batched=True, num_proc=10)
    dataset = dataset.filter(
        lambda x: x['content_no_comment'] != "" \
            and x['content_scrambled'] != "" \
            and x['content_no_comment_scrambled'] != "" \
            and x['content_randomized'] != "" \
            and x['content_no_comment_randomized'] != "" \
            and x['content_randomized_w_dep'] != "" \
            and x['content_no_comment_randomized_w_dep'] != ""
            )
    
    num_samples_after = len(dataset)
    print(num_samples_after)
    print(f"Removed {num_samples_before - num_samples_after} samples.")
    
    dataset.save_to_disk("/home/uchiyama.fumiya/ucllm/ucllm/program/outputs/data/python_ablation_500000_v3")
    dataset.push_to_hub('fumiyau/python_ablation_500000', private=True)