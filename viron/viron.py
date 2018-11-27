from flask import Flask, jsonify
import yaml
app = Flask(__name__)

@app.route('/')
def route():
    return 'Hello World!'

@app.route('/viron_authtype')
def auth_type():
    l = [
        {'type': 'email', 'provider': 'viron-example',
            'url': '/signin', 'method': 'POST'},
        {'type': 'signout', 'provider': '', 'url': '/signout', 'method': 'POST'}
    ]
    return jsonify(l)


@app.route('/swagger.json')
def swagger():
    with open('./swagger.yaml') as f:
        return jsonify(yaml.load(f))


@app.route('/viron')
def show():
    result = {
        # エンドポイントに関する情報
        'theme': 'standard',  # カラーテーマ
        'color': 'white',  # ドットカラー
        'name': 'Viron example - local',  # エンドポイントの名称。サービス名や環境など
        'tags': [  # エンドポイントに付与するタグ
            'local',
            'viron',
            'example'
        ],
        'thumbnail': 'https:#avatars3.githubusercontent.com/u/23251378?v=3&s=200',  # サムネイル画像URL

        # グローバルメニュー
        'pages': [
            {
                'section': 'dashboard',  # 大項目。'dashboard' or 'manage'
                'group': '',  # 中項目。空の場合はsection直下にcomponentsを配置
                'id': 'quickview',  # ページのID。全ページでユニークになっている必要があります
                'name': 'クイックビュー',  # ページ名
                'components': [  # メニューからページを選択した際に表示されるコンポーネントの一覧
                    {
                        'api': {  # コンポーネントに表示する値を取得するためのAPI
                            'method': 'get',
                            'path': '/stats/dau'
                        },
                        'name': 'DAU',  # コンポーネント名
                        # コンポーネントスタイル。数字(number)、テーブル(table)の他に各種グラフ(graph-*)が利用できます
                        'style': 'number'
                    },
                    {
                        'api': {
                            'method': 'get',
                            'path': '/stats/mau'
                        },
                        'name': 'MAU',
                        'style': 'number'
                    },
                ]
            },
            {
                'section': 'manage',
                'group': 'user',
                'id': 'user',
                'name': 'ユーザー',
                'components': [
                    {
                        'api': {
                            'method': 'get',
                            'path': '/user'
                        },
                        'name': 'ユーザー',
                        'style': 'table',
                        'primary': 'id',  # テーブルデータの主キーにあたるフィールド
                        'pagination': True,  # テーブルスタイルのページャーを有効にするフラグ
                        'query': [  # テーブルスタイルの検索フィールドを指定
                            {
                                'key': 'name',
                                'type': 'string'
                            }
                        ],
                        'table_labels': [  # テーブルスタイルの見出しにするフィールドを指定
                            'id',
                            'name'
                        ],
                        'actions': [  # 他APIの関連付けが必要な場合は指定。詳細は下記[API間の関連付け]を参照
                            '/user/download/csv'
                        ]
                    }
                ]
            }
        ]
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
