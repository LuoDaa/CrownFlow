import pytest
from fastapi import HTTPException
from app import config
from app.db import init_db,db
from app.api import routes
from app.models import CaseCreate


@pytest.fixture
def case_store(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.sqlite')
    monkeypatch.setattr(routes,'CASE_DIR',tmp_path/'cases')
    init_db()
    case=routes.create_case(CaseCreate(name='delete test'))['id']
    other=routes.create_case(CaseCreate(name='keep test'))['id']
    directory=routes.CASE_DIR/case
    (directory/'scan.stl').write_bytes(b'test mesh')
    (routes.CASE_DIR/other/'keep.stl').write_bytes(b'keep')
    with db() as c:
        c.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?)',('asset',case,'lower',str(directory/'scan.stl'),'scan.stl','{}','now'))
        c.execute('INSERT INTO designs(id,case_id,target_fdi,support_type,parameters_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',('design',case,46,'prep','{}','now','now'))
        c.execute('INSERT INTO design_revisions(id,design_id,revision_no,kind,payload_json,created_at) VALUES(?,?,?,?,?,?)',('rev','design',1,'outer','{}','now'))
    return case,other,directory


def test_delete_cascades_and_removes_files(case_store):
    case,other,directory=case_store
    assert routes.delete_case(case)=={'deleted':case}
    assert not directory.exists()
    assert (routes.CASE_DIR/other/'keep.stl').read_bytes()==b'keep'
    with db() as c:
        for table in ('assets','designs','design_revisions'):
            assert c.execute(f'SELECT count(*) FROM {table}').fetchone()[0]==0
    assert routes.case_exists(other) and not routes.case_exists(case)
    with pytest.raises(HTTPException) as error:routes.delete_case(case)
    assert error.value.status_code==404


def test_file_failure_does_not_claim_deleted(case_store,monkeypatch):
    case,_,directory=case_store
    def fail(_):raise PermissionError('locked')
    monkeypatch.setattr(routes.shutil,'rmtree',fail)
    with pytest.raises(HTTPException) as error:routes.delete_case(case)
    assert error.value.status_code==500
    assert routes.case_exists(case) and (directory/'scan.stl').exists()
    with db() as c:assert c.execute('SELECT count(*) FROM designs').fetchone()[0]==1


def test_rejects_path_escape(case_store):
    with pytest.raises(HTTPException) as error:routes.delete_case('..')
    assert error.value.status_code==400
